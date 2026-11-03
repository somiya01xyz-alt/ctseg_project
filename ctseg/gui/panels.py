"""
Dock panels for the napari front end.

Every panel is a thin wrapper: it reads what the user set, calls a function from
the ctseg library, and writes the result back to napari's Labels layer through
`Ctx.commit`, so every button can be undone with Ctrl+Z.

    Select   : threshold (live preview), flood fill (Ctrl+F), connected components
    Brush    : round / square, 2D / 3D brush and eraser (stamp or drag), size
    Edit     : morphology, propagate, interpolate, thin structures (outline / interface)
    ROI      : rectangle / ellipse / polygon / freehand ROIs -> label, stats, save, load
    Labels   : label table, project save / load, Dice against a reference folder
    View     : 2D / 3D switch, show-only-active-label, STL mesh export
    Help     : keyboard shortcuts
"""
import glob
import os
from pathlib import Path

import numpy as np
import tifffile
from magicgui import magicgui
from magicgui.widgets import ComboBox, Container, Label, LineEdit, PushButton

from ..core.label_table import LabelTable
from ..core.labelmap import LabelMap
from ..evaluate import dice
from ..fileio.files import _natural_key, load_project, save_project
from ..tools.brush import stroke_mask
from ..tools.morphology import apply_morphology, interface, outline
from ..tools.propagate import interpolate_between_slices, propagate_slices
from ..tools.roi import ROI, load_rois, save_rois
from ..tools.mesh import export_label_stl
from ..tools.select import (apply_thresholds, component_at, flood_fill, largest_component, multi_otsu_thresholds,
                            multi_seed_grow, region_grow_auto, remove_small)


# ── shared context ───────────────────────────────────────────────────────────

class Ctx:
    """Everything the panels need: the viewer, the CT volume, the Labels layer."""

    def __init__(self, viewer, volume, labels_layer, downsample=1):
        self.viewer = viewer
        self.volume = volume
        self.labels = labels_layer
        self.table = LabelTable()
        self.ds = downsample
        self.roi_layer = None
        self.refresh_labels = lambda: None      # set by label_panel
        self.image_layer = None
        self.preview_layer = None               # set by threshold_panel
        self.seeds = []                         # (label, z, y, x), used by the multi-seed panel
        self.seeds_layer = None
        self.otsu_cache = {}                    # classes -> thresholds, so a slice and the volume agree

    # -- where the user is looking
    @property
    def z(self):
        return int(self.viewer.dims.current_step[0])

    def z_end(self, n_more):
        return min(self.z + int(n_more), self.volume.shape[0] - 1)

    def cursor_voxel(self):
        pos = self.viewer.cursor.position
        z, y, x = (int(round(v)) for v in pos)
        return (max(0, min(z, self.volume.shape[0] - 1)),
                max(0, min(y, self.volume.shape[1] - 1)),
                max(0, min(x, self.volume.shape[2] - 1)))

    # -- writing to the Labels layer (undoable)
    def commit(self, z0, new_block):
        """Write slices z0.. into the Labels layer through napari's history.
        Only voxels that differ are written. Returns how many changed."""
        z1 = z0 + new_block.shape[0]
        old = self.labels.data[z0:z1]
        changed = old != new_block
        if not changed.any():
            return 0
        iz, iy, ix = np.nonzero(changed)
        self.labels.data_setitem((iz + z0, iy, ix), new_block[changed])
        return int(changed.sum())

    def block_lm(self, z0, z1):
        """A small LabelMap holding only slices z0..z1 (inclusive)."""
        lm = LabelMap((z1 - z0 + 1,) + self.volume.shape[1:], dtype=self.labels.data.dtype)
        lm.data[...] = self.labels.data[z0:z1 + 1]
        return lm

    def say(self, text):
        print(text)
        try:
            self.viewer.status = text
        except Exception:
            pass


# ── Select: threshold with live preview ──────────────────────────────────────

def threshold_panel(ctx):
    vmin, vmax = float(ctx.volume.min()), float(ctx.volume.max())
    preview = ctx.viewer.add_labels(np.zeros(ctx.volume.shape[1:], np.uint8),
                                    name="threshold preview", opacity=0.35)

    @magicgui(call_button="Apply threshold to active label",
              low={"widget_type": "FloatSlider", "min": vmin, "max": vmax},
              high={"widget_type": "FloatSlider", "min": vmin, "max": vmax},
              n_more_slices={"min": 0, "max": 100})
    def threshold_widget(low: float = vmin + 0.4 * (vmax - vmin),
                         high: float = vmax, n_more_slices: int = 0):
        z, z1 = ctx.z, ctx.z_end(n_more_slices)
        vol = ctx.volume[z:z1 + 1]
        block = ctx.labels.data[z:z1 + 1].copy()
        block[(vol >= low) & (vol <= high)] = ctx.labels.selected_label
        n = ctx.commit(z, block)
        ctx.say(f"threshold {low:.0f}-{high:.0f}: {n} voxels changed on slices {z}-{z1} (Ctrl+Z to undo)")

    def update_preview(*_):
        img = ctx.volume[ctx.z]
        m = (img >= threshold_widget.low.value) & (img <= threshold_widget.high.value)
        preview.data = m.astype(np.uint8)

    ctx.preview_layer = preview
    threshold_widget.low.changed.connect(update_preview)
    threshold_widget.high.changed.connect(update_preview)
    try:
        ctx.viewer.dims.events.current_step.connect(update_preview)
    except Exception:
        pass
    update_preview()
    threshold_widget.update_preview = update_preview
    return threshold_widget


# ── Select: connected components ─────────────────────────────────────────────

def components_panel(ctx):
    @magicgui(call_button="Keep largest component (this slice)")
    def largest_widget():
        z, label = ctx.z, ctx.labels.selected_label
        block = ctx.labels.data[z:z + 1].copy()
        mask = block[0] == label
        if not mask.any():
            return ctx.say("nothing to keep: the active label is empty on this slice")
        keep = largest_component(mask)
        block[0][mask & ~keep] = 0
        ctx.say(f"kept the largest component, removed {ctx.commit(z, block)} voxels")

    @magicgui(call_button="Remove small objects (this slice)")
    def small_widget(min_size: int = 50):
        z, label = ctx.z, ctx.labels.selected_label
        block = ctx.labels.data[z:z + 1].copy()
        mask = block[0] == label
        keep = remove_small(mask, min_size)
        block[0][mask & ~keep] = 0
        ctx.say(f"removed {ctx.commit(z, block)} voxels in objects under {min_size} px")

    def keep_under_cursor(viewer):
        """Ctrl+K: keep only the connected piece of the active label under the cursor."""
        z, y, x = ctx.cursor_voxel()
        label = ctx.labels.selected_label
        block = ctx.labels.data[z:z + 1].copy()
        mask = block[0] == label
        comp = component_at(mask, (y, x))
        if not comp.any():
            return ctx.say("no active-label region under the cursor")
        block[0][mask & ~comp] = 0
        ctx.say(f"kept the component under the cursor, removed {ctx.commit(z, block)} voxels")

    ctx.viewer.bind_key("Ctrl-K")(keep_under_cursor)
    box = Container(widgets=[largest_widget, small_widget], labels=False)
    box.largest, box.small, box.keep_under_cursor = largest_widget, small_widget, keep_under_cursor
    return box


# ── Select: multi-Otsu (several materials in one click) ──────────────────────

def ensure_labels(ctx, ids, names=None):
    """Make sure the label table has an entry for every id (so each class gets a name and colour)."""
    added = False
    for i in ids:
        if i not in ctx.table.entries:
            ctx.table.add((names or {}).get(i, f"Class {i}"), label_id=int(i))
            added = True
    if added:
        ctx.refresh_labels()


def multi_otsu_panel(ctx):
    @magicgui(call_button="Apply multi-Otsu",
              scope={"choices": ["this slice", "whole volume"]})
    def otsu_widget(classes: int = 4, scope: str = "this slice", protect_other_labels: bool = True,
                    sample_step: int = 8):
        """Class 0 (darkest) stays background; class 1..classes-1 become labels 1..classes-1.
        Thresholds are always computed on a subsample of the whole volume, so 'this slice' and
        'whole volume' use the same numbers."""
        classes = int(classes)
        key = (classes, int(sample_step))
        if key not in ctx.otsu_cache:
            ctx.otsu_cache[key] = multi_otsu_thresholds(ctx.volume, classes, sample_step)
        th = ctx.otsu_cache[key]
        z, z1 = (ctx.z, ctx.z) if scope == "this slice" else (0, ctx.volume.shape[0] - 1)
        cls = apply_thresholds(ctx.volume[z:z1 + 1], th)
        ensure_labels(ctx, range(1, classes))
        block = ctx.labels.data[z:z1 + 1].copy()
        for c in range(1, classes):
            sel = cls == c
            if protect_other_labels:
                sel &= block == 0
            block[sel] = c
        n = ctx.commit(z, block)
        ctx.say(f"multi-Otsu, {classes} classes, thresholds {', '.join(f'{t:.0f}' for t in th)}: "
                f"{n} voxels changed on slices {z}-{z1} (Ctrl+Z to undo)")

    return otsu_widget


# ── Select: several seeds, each with its own label, grown together ───────────

def seeds_panel(ctx):
    def redraw():
        layer = ctx.seeds_layer
        if layer is None:
            return
        pts = np.array([s[1:] for s in ctx.seeds], float).reshape(-1, 3)
        cols = []
        for s in ctx.seeds:
            e = ctx.table.entries.get(s[0])
            cols.append(tuple(c / 255 for c in e["color"]) + (1.0,) if e else (1, 1, 1, 1))
        layer.data = pts
        if len(pts):
            layer.face_color = cols

    if hasattr(ctx.viewer, "add_points"):
        ctx.seeds_layer = ctx.viewer.add_points(np.zeros((0, 3)), name="seeds", ndim=3, size=10,
                                                 border_color="white")

    def add_seed(viewer=None):
        z, y, x = ctx.cursor_voxel()
        label = int(ctx.labels.selected_label)
        ensure_labels(ctx, [label])
        ctx.seeds.append((label, z, y, x))
        redraw()
        ctx.say(f"seed {len(ctx.seeds)}: label {label} at ({z},{y},{x})")

    @magicgui(call_button="Add seed at cursor (active label)")
    def add_widget():
        add_seed()

    @magicgui(call_button="Clear seeds")
    def clear_widget():
        ctx.seeds.clear()
        redraw()
        ctx.say("seeds cleared")

    @magicgui(call_button="Grow all seeds", scope={"choices": ["seed slices only (2D)", "3D slab around seeds"]})
    def grow_widget(k: float = 4.0, radius: int = 8, scope: str = "seed slices only (2D)",
                    slab_margin: int = 3, protect_other_labels: bool = True):
        if not ctx.seeds:
            return ctx.say("no seeds yet: put the cursor on a material, pick its label, then 'Add seed'")
        zmax = ctx.volume.shape[0] - 1
        if scope.startswith("seed"):
            groups = {}
            for lab, z, y, x in ctx.seeds:
                groups.setdefault(z, []).append((lab, (y, x)))
            jobs = [(z, z, g) for z, g in groups.items()]
        else:
            z0 = max(min(s[1] for s in ctx.seeds) - slab_margin, 0)
            z1 = min(max(s[1] for s in ctx.seeds) + slab_margin, zmax)
            jobs = [(z0, z1, [(lab, (z - z0, y, x)) for lab, z, y, x in ctx.seeds])]
        total = 0
        for z0, z1, seeds in jobs:
            img = ctx.volume[z0:z1 + 1]
            img = img[0] if z0 == z1 and scope.startswith("seed") else img
            grown = multi_seed_grow(img, seeds, k=k, radius=int(radius))
            grown = grown.reshape((z1 - z0 + 1,) + ctx.volume.shape[1:])
            block = ctx.labels.data[z0:z1 + 1].copy()
            sel = grown > 0
            if protect_other_labels:
                sel &= (block == 0) | (block == grown)
            block[sel] = grown[sel]
            total += ctx.commit(z0, block)
        ctx.say(f"grew {len(ctx.seeds)} seed(s) with {len(set(s[0] for s in ctx.seeds))} label(s): "
                f"{total} voxels changed (Ctrl+Z to undo)")

    ctx.viewer.bind_key("Ctrl-J")(add_seed)
    hint = Label(value="Click a material, pick its label in the Labels tab, press Ctrl+J (or the "
                       "button). Repeat for each material, then 'Grow all seeds'.")
    box = Container(widgets=[hint, add_widget, clear_widget, grow_widget], labels=False)
    box.add, box.clear, box.grow, box.add_seed = add_widget, clear_widget, grow_widget, add_seed
    return box


# ── View: 3D, show-only-active-label, STL export ─────────────────────────────

BIG_VOLUME = 60_000_000      # voxels; beyond this 3D rendering gets slow


def view_panel(ctx):
    def set_3d(on):
        ctx.viewer.dims.ndisplay = 3 if on else 2
        try:
            if ctx.preview_layer is not None:
                ctx.preview_layer.visible = not on      # the 2D threshold preview makes no sense in 3D
            ctx.labels.rendering = "iso_categorical" if on else "translucent"
        except Exception:
            pass
        if on and ctx.volume.size > BIG_VOLUME:
            ctx.say(f"3D with {ctx.volume.size / 1e6:.0f} M voxels may lag: restart with a larger "
                    f"downsample (e.g. 2 or 4), or hide 'CT volume' to show only the labels")
        else:
            ctx.say("3D view" if on else "2D view")

    @magicgui(call_button="Switch 2D / 3D")
    def toggle_widget():
        set_3d(ctx.viewer.dims.ndisplay != 3)

    @magicgui(call_button="Apply")
    def display_widget(show_ct_in_3d: bool = False, show_only_active_label: bool = False):
        try:
            if ctx.image_layer is not None:
                ctx.image_layer.visible = show_ct_in_3d or ctx.viewer.dims.ndisplay != 3
            ctx.labels.show_selected_label = show_only_active_label
        except Exception:
            pass

    @magicgui(call_button="Export active label as STL", path={"mode": "w", "filter": "*.stl"})
    def export_widget(path: Path = Path("label.stl"), step_size: int = 1):
        """step_size 2 or 3 gives a coarser, smaller mesh. Sizes are in full-resolution voxels."""
        label = int(ctx.labels.selected_label)
        ds = float(ctx.ds)
        try:
            n = export_label_stl(ctx.labels.data, label, str(path), (ds, ds, ds), step_size)
        except ValueError as e:
            return ctx.say(str(e))
        ctx.say(f"label {label}: {n} triangles written to {path}")

    box = Container(widgets=[toggle_widget, display_widget, export_widget], labels=False)
    box.toggle, box.display, box.export, box.set_3d = toggle_widget, display_widget, export_widget, set_3d
    return box


# ── Select: flood fill (paint bucket by intensity) ───────────────────────────

def flood_panel(ctx):
    @magicgui(call_button="Flood fill at cursor",
              scope={"choices": ["this slice", "this + next N slices (3D)"]})
    def flood_widget(tolerance: float = 20.0, scope: str = "this slice",
                     n_more_slices: int = 3):
        z, y, x = ctx.cursor_voxel()
        label = ctx.labels.selected_label
        one = scope == "this slice"
        z1 = z if one else ctx.z_end(n_more_slices)
        img = ctx.volume[z:z1 + 1].astype(float)
        m = flood_fill(img[0], (y, x), tolerance) if one else flood_fill(img, (0, y, x), tolerance)
        block = ctx.labels.data[z:z1 + 1].copy()
        block[m.reshape(block.shape)] = label
        n = ctx.commit(z, block)
        ctx.say(f"flood fill @ ({z},{y},{x}) tol={tolerance:g} label={label}: {n} voxels (Ctrl+Z to undo)")

    ctx.viewer.bind_key("Ctrl-F")(lambda viewer: flood_widget())
    return flood_widget


# ── Brush: round / square, 2D / 3D, paint or erase ───────────────────────────

def apply_stroke(ctx, points, radius, shape="round", mode="2d", action="paint",
                 protect=False, only_selected=False):
    """Paint or erase a stroke (list of (z, y, x) voxels) with the library's brush stamp.
    The whole stroke is one undo step. Returns the number of voxels changed."""
    mask, origin = stroke_mask(points, radius, shape, mode)
    vz, vy, vx = ctx.labels.data.shape
    lo = [max(o, 0) for o in origin]
    hi = [min(o + n, m) for o, n, m in zip(origin, mask.shape, (vz, vy, vx))]
    if any(h <= l for l, h in zip(lo, hi)):
        return 0
    m = mask[tuple(slice(l - o, h - o) for l, h, o in zip(lo, hi, origin))]
    box = tuple(slice(l, h) for l, h in zip(lo, hi))
    block = ctx.labels.data[box].copy()
    label = ctx.labels.selected_label
    if action == "paint":
        sel = m & ((block == 0) | (block == label)) if protect else m
        block[sel] = label
    else:
        sel = m & (block == label) if only_selected else m
        block[sel] = 0
    old = ctx.labels.data[box]
    changed = old != block
    if not changed.any():
        return 0
    iz, iy, ix = np.nonzero(changed)
    ctx.labels.data_setitem((iz + lo[0], iy + lo[1], ix + lo[2]), block[changed])
    return int(changed.sum())


def brush_panel(ctx):
    @magicgui(call_button="Stamp at cursor",
              shape={"choices": ["round", "square"]}, mode={"choices": ["2d", "3d"]},
              action={"choices": ["paint", "erase"]})
    def brush_widget(shape: str = "round", mode: str = "2d", radius: int = 8,
                     action: str = "paint", protect_other_labels: bool = False,
                     erase_only_selected_label: bool = False,
                     drag_with_ctseg_brush: bool = False):
        z, y, x = ctx.cursor_voxel()
        n = apply_stroke(ctx, [(z, y, x)], radius, shape, mode, action,
                         protect_other_labels, erase_only_selected_label)
        ctx.say(f"{action} {shape} {mode} brush r={radius} @ ({z},{y},{x}): {n} voxels (Ctrl+Z to undo)")

    # Optional: drag with our brush (square / 3D shapes napari does not have).
    # While it is on, napari's own pan is switched off, so the mouse draws instead of moving.
    def drag(layer, event):
        if not brush_widget.drag_with_ctseg_brush.value:
            return
        pts = []

        def add(ev):
            p = layer.world_to_data(ev.position)
            pts.append(tuple(int(round(v)) for v in p[-3:]))
        add(event)
        yield
        while event.type == "mouse_move":
            add(event)
            yield
        if pts:
            n = apply_stroke(ctx, pts, brush_widget.radius.value, brush_widget.shape.value,
                             brush_widget.mode.value, brush_widget.action.value,
                             brush_widget.protect_other_labels.value,
                             brush_widget.erase_only_selected_label.value)
            ctx.say(f"stroke of {len(pts)} points: {n} voxels (Ctrl+Z to undo)")

    def toggle(on):
        try:
            ctx.viewer.camera.mouse_pan = not on
            ctx.labels.mode = "pan_zoom"
        except Exception:
            pass

    try:
        ctx.labels.mouse_drag_callbacks.append(drag)
        brush_widget.drag_with_ctseg_brush.changed.connect(toggle)
    except Exception:
        pass
    brush_widget.drag = drag
    return brush_widget


# ── Edit: morphology, propagate, interpolate ─────────────────────────────────

def edit_panel(ctx):
    @magicgui(call_button="Apply morphology",
              op={"choices": ["dilate", "erode", "open", "close", "fill"]},
              scope={"choices": ["this slice", "this + next N slices (3D)"]})
    def morph_widget(op: str = "close", radius: int = 2,
                     scope: str = "this slice", n_more_slices: int = 3):
        z, label = ctx.z, ctx.labels.selected_label
        one = scope == "this slice"
        z1 = z if one else ctx.z_end(n_more_slices)
        lm = ctx.block_lm(z, z1)
        apply_morphology(lm, label, op, radius, z=0 if one else None)
        n = ctx.commit(z, lm.data)
        ctx.say(f"{op} radius {radius} ({scope}): {n} voxels changed (Ctrl+Z to undo)")

    @magicgui(call_button="Propagate label forward N slices")
    def propagate_widget(n_slices: int = 5, grow: int = 3, min_size: int = 50):
        z, label = ctx.z, ctx.labels.selected_label
        z_end = ctx.z_end(n_slices)
        lm = ctx.block_lm(z, z_end)
        done = propagate_slices(lm, ctx.volume[z:z_end + 1], label, 0, z_end - z,
                                grow=grow, min_size=min_size)
        ctx.commit(z, lm.data)
        ctx.say(f"propagated {done} slices from {z} toward {z_end} (Ctrl+Z to undo)")

    @magicgui(call_button="Interpolate between this slice and slice #")
    def interpolate_widget(other_slice: int = 0):
        z, label = ctx.z, ctx.labels.selected_label
        z0, z1 = min(z, other_slice), max(z, other_slice)
        lm = ctx.block_lm(z0, z1)
        ok = interpolate_between_slices(lm, label, 0, z1 - z0)
        ctx.commit(z0, lm.data)
        ctx.say("interpolated" if ok else "nothing between those slices")

    @magicgui(call_button="Make thin structure from labels",
              kind={"choices": ["outline", "interface"]},
              scope={"choices": ["this slice", "this + next N slices"]})
    def thin_widget(kind: str = "outline", label_a: int = 1, label_b: int = 2,
                    thickness: int = 1, scope: str = "this slice", n_more_slices: int = 3):
        """outline: edge of A (and B, merged). interface: where A touches B.
        The result is painted with the active label (use a new label, e.g. Braze)."""
        z, target = ctx.z, ctx.labels.selected_label
        z1 = z if scope == "this slice" else ctx.z_end(n_more_slices)
        block = ctx.labels.data[z:z1 + 1].copy()
        a, b = block == label_a, block == label_b
        m = outline(a | b, thickness) if kind == "outline" else interface(a, b, thickness)
        block[m] = target
        n = ctx.commit(z, block)
        ctx.say(f"{kind} (A={label_a}, B={label_b}, thickness {thickness}) -> label {target}: {n} voxels")

    box = Container(widgets=[morph_widget, propagate_widget, interpolate_widget, thin_widget],
                    labels=False)
    box.morph, box.propagate, box.interpolate = morph_widget, propagate_widget, interpolate_widget
    box.thin = thin_widget
    return box


# ── ROI ──────────────────────────────────────────────────────────────────────

_KIND = {"rectangle": "rectangle", "ellipse": "ellipse", "polygon": "polygon",
         "path": "freehand", "line": "freehand"}


def shapes_to_rois(layer):
    rois = []
    for verts, stype in zip(layer.data, layer.shape_type):
        v = np.asarray(verts, float)
        z = int(round(v[:, 0].mean()))
        pts = [(float(y), float(x)) for _, y, x in v]
        rois.append(ROI(kind=_KIND.get(stype, "polygon"), z=z, points=pts,
                        name=f"roi{len(rois) + 1}"))
    return rois


def rois_to_shapes(layer, rois):
    for r in rois:
        data = np.array([[r.z, y, x] for y, x in r.points], float)
        if r.kind in ("rectangle", "ellipse") and len(data) == 2:
            (y0, x0), (y1, x1) = r.points
            data = np.array([[r.z, y0, x0], [r.z, y0, x1], [r.z, y1, x1], [r.z, y1, x0]], float)
        layer.add(data, shape_type="polygon" if r.kind == "freehand" else r.kind)


def roi_panel(ctx):
    layer = ctx.viewer.add_shapes(ndim=3, name="ROIs", edge_color="yellow",
                                  face_color="transparent", edge_width=2)
    ctx.roi_layer = layer

    def rois_here():
        return [r for r in shapes_to_rois(layer) if r.z == ctx.z]

    @magicgui(call_button="Apply ROIs on this slice to active label",
              action={"choices": ["add to label", "remove from label"]})
    def apply_widget(action: str = "add to label"):
        z, label = ctx.z, ctx.labels.selected_label
        rois = rois_here()
        if not rois:
            return ctx.say("no ROI drawn on this slice (select the 'ROIs' layer and draw one)")
        m = np.zeros(ctx.volume.shape[1:], bool)
        for r in rois:
            m |= r.to_mask(m.shape)
        block = ctx.labels.data[z:z + 1].copy()
        if action == "add to label":
            block[0][m] = label
        else:
            block[0][m & (block[0] == label)] = 0
        ctx.say(f"{len(rois)} ROI(s): {ctx.commit(z, block)} voxels changed (Ctrl+Z to undo)")

    stats_text = Label(value="ROI statistics appear here")

    @magicgui(call_button="ROI statistics (this slice)")
    def stats_widget():
        rois = rois_here()
        if not rois:
            stats_text.value = "no ROI on this slice"
            return
        lines = []
        for r in rois:
            s = r.stats(ctx.volume[ctx.z])
            lines.append(f"{r.name} ({r.kind}): area {s['area']} px"
                         + (f", mean {s['mean']:.1f}, std {s['std']:.1f}, "
                            f"min {s['min']:.0f}, max {s['max']:.0f}" if s["area"] else ""))
        stats_text.value = "\n".join(lines)

    @magicgui(call_button="Save all ROIs", path={"mode": "w", "filter": "*.json"})
    def save_widget(path: Path = Path("rois.json")):
        save_rois(shapes_to_rois(layer), str(path))
        ctx.say(f"saved {len(layer.data)} ROI(s) to {path}")

    @magicgui(call_button="Load ROIs", path={"mode": "r", "filter": "*.json"})
    def load_widget(path: Path = Path("rois.json")):
        rois = load_rois(str(path))
        rois_to_shapes(layer, rois)
        ctx.say(f"loaded {len(rois)} ROI(s) from {path}")

    hint = Label(value="Select the 'ROIs' layer, pick the rectangle, ellipse, polygon or path\n"
                       "tool, and draw on a slice. Then use the buttons below.")
    box = Container(widgets=[hint, apply_widget, stats_widget, stats_text, save_widget, load_widget],
                    labels=False)
    box.apply, box.stats, box.stats_text, box.save, box.load = (
        apply_widget, stats_widget, stats_text, save_widget, load_widget)
    box.layer = layer
    return box


# ── Labels, project, evaluation ──────────────────────────────────────────────

def label_panel(ctx):
    selector = ComboBox(label="Active label", choices=[])
    name_edit = LineEdit(label="New label", value="")
    add_btn = PushButton(text="Add label")
    std_btn = PushButton(text="Add Ceramic, Metal, Braze, Outer")

    def apply_colors():
        cols = {i: tuple(c / 255 for c in e["color"]) + (1.0,) for i, e in ctx.table.entries.items()}
        try:
            from napari.utils.colormaps import DirectLabelColormap
            ctx.labels.colormap = DirectLabelColormap(color_dict={None: (0, 0, 0, 0), 0: (0, 0, 0, 0), **cols})
        except Exception:
            try:
                ctx.labels.color = {None: (0, 0, 0, 0), 0: (0, 0, 0, 0), **cols}
            except Exception:
                pass

    def refresh():
        choices = [(f"{i}: {e['name']}", i) for i, e in sorted(ctx.table.entries.items())]
        selector.choices = choices
        apply_colors()

    ctx.refresh_labels = refresh

    def add_one(name):
        i = ctx.table.add(name)
        refresh()
        selector.value = i
        ctx.labels.selected_label = i
        return i

    def on_add():
        add_one(name_edit.value.strip() or f"label {len(ctx.table.entries) + 1}")
        name_edit.value = ""

    def on_standard():
        for n in ["Ceramic", "Metal", "Braze", "Outer"]:
            add_one(n)

    def on_select(value):
        if value:
            ctx.labels.selected_label = int(value)

    add_btn.changed.connect(on_add)
    std_btn.changed.connect(on_standard)
    selector.changed.connect(on_select)

    # project save / load
    @magicgui(call_button="Save project (labels + label table + ROIs)", folder={"mode": "d"})
    def save_widget(folder: Path = Path("ctseg_project_out")):
        lm = LabelMap((1, 1, 1), dtype=ctx.labels.data.dtype)
        lm.data = ctx.labels.data                       # no copy
        rois = shapes_to_rois(ctx.roi_layer) if ctx.roi_layer is not None else []
        save_project(str(folder), lm, ctx.table, rois)
        ctx.say(f"project saved to {folder}")

    @magicgui(call_button="Load project", folder={"mode": "d"})
    def load_widget(folder: Path = Path("ctseg_project_out")):
        lm, table, rois = load_project(str(folder))
        if lm.shape != ctx.labels.data.shape:
            return ctx.say(f"project shape {lm.shape} does not match the open volume "
                           f"{ctx.labels.data.shape} (was it saved at the same downsample?)")
        ctx.labels.data = lm.data
        ctx.table.entries = table.entries
        refresh()
        if ctx.roi_layer is not None and rois:
            rois_to_shapes(ctx.roi_layer, rois)
        ctx.say(f"project loaded from {folder}")

    # evaluation against a reference mask folder
    @magicgui(call_button="Dice vs reference folder (this slice, active label)", folder={"mode": "d"})
    def eval_widget(folder: Path = Path("."), background: int = 18):
        files = sorted(glob.glob(os.path.join(str(folder), "*.tif")), key=_natural_key)
        if not files:
            return ctx.say(f"no .tif files in {folder}")
        z = ctx.z
        if z >= len(files):
            return ctx.say(f"reference folder has only {len(files)} slices")
        ref = tifffile.imread(files[z])
        ref = ref[..., 0] if ref.ndim == 3 else ref
        ref = ref[::ctx.ds, ::ctx.ds] > background
        pred = ctx.labels.data[z] == ctx.labels.selected_label
        if ref.shape != pred.shape:
            return ctx.say(f"reference {ref.shape} and labels {pred.shape} differ in size")
        ctx.say(f"slice {z}: Dice = {dice(pred, ref):.3f}")

    box = Container(widgets=[selector, name_edit, add_btn, std_btn, save_widget, load_widget, eval_widget],
                    labels=True)
    box.selector, box.add_btn, box.std_btn = selector, add_btn, std_btn
    box.name_edit, box.save, box.load, box.evaluate = name_edit, save_widget, load_widget, eval_widget
    box.add_one = add_one
    return box


HELP = """\
Painting (napari Labels layer, select the 'labels' layer first)
  2 = brush    1 = eraser    3 = bucket fill    4 = pick label
  [ and ] change brush size    n = new label    Ctrl+Z / Ctrl+Shift+Z = undo / redo
  'n edit dim' = 3 paints in 3D (spherical brush)
Region growing and selection
  Ctrl+G = grow a region from the cursor    Ctrl+F = flood fill at the cursor
  Ctrl+K = keep component under cursor
Tabs on the right
  Select : threshold with live preview, flood fill, connected components
  Brush  : round / square, 2D / 3D brush and eraser; tick 'drag' to draw with it
  Edit   : dilate, erode, open, close, fill, propagate, interpolate, thin structures
  ROI    : draw rectangle / ellipse / polygon / freehand ROIs, apply, statistics, save
  Labels : label names and colours, save / load project, Dice vs reference
  View   : 2D / 3D switch, show only the active label, export a label as an STL mesh
  Multi-seed (Select tab): Ctrl+J adds a seed with the active label, then 'Grow all seeds'
"""


def help_panel(ctx):
    return Container(widgets=[Label(value=HELP)], labels=False)
