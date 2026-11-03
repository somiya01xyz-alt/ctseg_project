"""
Tests for the napari panels. They use a small stand-in for napari's Viewer (the real one
needs OpenGL), but real napari Labels / Shapes layers and real magicgui widgets, so
the undo history and the panel logic are exercised for real.
Skipped automatically when napari / magicgui are not installed.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("magicgui")
tifffile = pytest.importorskip("tifffile")

from napari.layers import Image, Labels, Shapes   # noqa: E402


class _Signal:
    def connect(self, *_a, **_k):
        pass


class _Events:
    current_step = _Signal()


class _Dims:
    current_step = (5, 0, 0)
    ndisplay = 2
    events = _Events()


class _Cursor:
    position = (5, 60, 35)


class _Window:
    def __init__(self):
        self.docks = {}

    def add_dock_widget(self, widget, area=None, name=None, tabify=False):
        self.docks[name] = widget


class FakeViewer:
    def __init__(self):
        self.layers, self.keys = {}, {}
        self.dims, self.cursor, self.window, self.status = _Dims(), _Cursor(), _Window(), ""

    def add_image(self, data, name=None, **k):
        self.layers[name] = Image(data, name=name)
        return self.layers[name]

    def add_labels(self, data, name=None, **k):
        self.layers[name] = Labels(data, name=name)
        return self.layers[name]

    def add_shapes(self, name=None, ndim=2, **k):
        self.layers[name] = Shapes(ndim=ndim, name=name)
        return self.layers[name]

    def add_points(self, data, name=None, **k):
        from napari.layers import Points
        self.layers[name] = Points(data, name=name, ndim=3)
        return self.layers[name]

    def bind_key(self, key):
        def deco(f):
            self.keys[key] = f
            return f
        return deco


@pytest.fixture
def app():
    from ctseg.gui.napari_widget import build
    Z, H, W = 12, 120, 140
    yy, xx = np.mgrid[:H, :W]
    r = np.hypot(yy - 60, xx - 70)
    ring = (r > 20) & (r < 50)      # wide enough that an 8-px seed patch stays inside it
    rng = np.random.default_rng(0)
    vol = np.stack([(ring * 150 + 30 + rng.normal(0, 3, (H, W))).clip(0, 255).astype(np.uint8)
                    for _ in range(Z)])
    v = FakeViewer()
    ctx = build(v, vol)
    ctx.ring = ring
    return v, ctx


def count(ctx, z=5):
    return int((ctx.labels.data[z] > 0).sum())


def test_region_grow_is_undoable(app):
    v, ctx = app
    v.keys["Ctrl-G"](v)
    assert count(ctx) > 2000
    ctx.labels.undo()
    assert count(ctx) == 0
    ctx.labels.redo()
    assert count(ctx) > 2000


def test_threshold_apply_and_undo(app):
    v, ctx = app
    t = ctx.panels["threshold"]
    t.high.value = t.high.max; t.low.value = 100
    t()
    assert count(ctx) == int(ctx.ring.sum()) or abs(count(ctx) - int(ctx.ring.sum())) < 20
    ctx.labels.undo()
    assert count(ctx) == 0


def test_morphology_ops(app):
    v, ctx = app
    v.keys["Ctrl-G"](v)
    n0 = count(ctx)
    m = ctx.panels["edit"].morph
    m.op.value, m.radius.value = "dilate", 3
    m()
    assert count(ctx) > n0
    ctx.labels.undo()
    assert count(ctx) == n0
    m.op.value = "erode"
    m()
    assert count(ctx) < n0


def test_components_keep_largest(app):
    v, ctx = app
    v.keys["Ctrl-G"](v)
    block = ctx.labels.data[5:6].copy()
    block[0, 2:6, 2:6] = 1                       # a stray speck
    ctx.commit(5, block)
    with_speck = count(ctx)
    ctx.panels["components"].largest()
    assert count(ctx) == with_speck - 16


def test_propagate_keeps_ring_hollow(app):
    v, ctx = app
    v.keys["Ctrl-G"](v)
    ctx.panels["edit"].propagate()
    assert count(ctx, 6) > 2000
    assert ctx.labels.data[8, 60, 70] == 0       # centre of the ring stays empty
    ctx.labels.undo()
    assert count(ctx, 6) == 0


def test_roi_apply_stats_save_load(app, tmp_path):
    v, ctx = app
    roi = ctx.panels["roi"]
    roi.layer.add(np.array([[5, 10, 10], [5, 10, 30], [5, 30, 30], [5, 30, 10]], float),
                  shape_type="rectangle")
    roi.layer.add(np.array([[5, 90, 90], [5, 90, 110], [5, 110, 100]], float),
                  shape_type="polygon")
    roi.apply()
    assert count(ctx) > 400
    ctx.labels.undo()
    assert count(ctx) == 0
    roi.stats()
    assert "rectangle" in roi.stats_text.value and "polygon" in roi.stats_text.value
    path = tmp_path / "rois.json"
    roi.save.path.value = path
    roi.save()
    n = len(roi.layer.data)
    roi.load.path.value = path
    roi.load()
    assert len(roi.layer.data) == 2 * n


def test_labels_and_project_round_trip(app, tmp_path):
    v, ctx = app
    lab = ctx.panels["labels"]
    lab.std_btn.changed.emit(None)               # adds the four standard materials
    names = [e["name"] for e in ctx.table.entries.values()]
    assert names[:2] == ["Ceramic", "Ceramic"] or "Metal" in names
    v.keys["Ctrl-G"](v)
    before = ctx.labels.data.copy()
    lab.save.folder.value = tmp_path / "proj"
    lab.save()
    ctx.labels.data[...] = 0
    lab.load.folder.value = tmp_path / "proj"
    lab.load()
    assert np.array_equal(ctx.labels.data, before)


def test_dice_against_reference_folder(app, tmp_path):
    v, ctx = app
    v.keys["Ctrl-G"](v)
    for z in range(ctx.volume.shape[0]):
        ref = np.where(ctx.ring, 200, 18).astype(np.uint8)
        tifffile.imwrite(tmp_path / f"m{z:03d}.tif", ref)
    ev = ctx.panels["labels"].evaluate
    ev.folder.value = tmp_path
    ev()
    assert "Dice = 1.000" in v.status or "Dice = 0.9" in v.status


def test_brush_square_3d_and_erase(app):
    v, ctx = app
    from ctseg.gui.panels import apply_stroke
    n = apply_stroke(ctx, [(5, 60, 70)], 3, "square", "3d")
    assert n == 7 ** 3
    assert ctx.labels.data[4, 60, 70] == 1 and ctx.labels.data[9, 60, 70] == 0   # spans slices
    ctx.labels.undo()
    assert count(ctx) == 0
    apply_stroke(ctx, [(5, 60, 60), (5, 60, 80)], 4, "round", "2d")
    assert count(ctx) > 0 and ctx.labels.data[4, 60, 70] == 0                    # one slice only
    n = apply_stroke(ctx, [(5, 60, 70)], 2, "square", "2d", action="erase")
    assert n == 25
    ctx.labels.undo()
    assert ctx.labels.data[5, 60, 70] == 1


def test_brush_protects_other_labels(app):
    v, ctx = app
    from ctseg.gui.panels import apply_stroke
    ctx.labels.selected_label = 2
    apply_stroke(ctx, [(5, 60, 70)], 3, "square", "2d")
    ctx.labels.selected_label = 1
    apply_stroke(ctx, [(5, 60, 70)], 5, "square", "2d", protect=True)
    assert (ctx.labels.data[5] == 2).sum() == 49          # label 2 untouched
    assert (ctx.labels.data[5] == 1).sum() == 121 - 49


def test_flood_fill_button(app):
    v, ctx = app
    f = ctx.panels["flood"]
    f.tolerance.value = 20
    f()
    assert abs(count(ctx) - int(ctx.ring.sum())) < 300     # fills the ring only
    ctx.labels.undo()
    assert count(ctx) == 0


def test_thin_outline_and_interface(app):
    v, ctx = app
    ctx.labels.data[5][ctx.ring] = 1
    ctx.labels.selected_label = 3
    t = ctx.panels["edit"].thin
    t.kind.value, t.label_a.value, t.label_b.value, t.thickness.value = "outline", 1, 0, 1
    t()
    n = int((ctx.labels.data[5] == 3).sum())
    assert 0 < n < 0.2 * ctx.ring.sum()                    # a thin line, not the whole ring
    ctx.labels.undo()
    assert (ctx.labels.data[5] == 3).sum() == 0


def test_library_outline_interface():
    from ctseg.tools.morphology import interface, outline
    m = np.zeros((20, 20), bool); m[5:15, 5:15] = True
    o = outline(m, 1)
    assert o.sum() == 36 and not o[8, 8]
    a = np.zeros((10, 10), bool); a[:, :5] = True
    b = np.zeros((10, 10), bool); b[:, 5:] = True
    band = interface(a, b, 1)
    assert band.sum() == 20 and band[:, 4:6].all()


def test_drag_callback_draws_one_undoable_stroke(app):
    v, ctx = app
    b = ctx.panels["brush"]
    b.drag_with_ctseg_brush.value = True
    b.shape.value, b.mode.value, b.radius.value = "square", "2d", 2

    class Ev:
        type = "mouse_press"
        position = (5, 60, 60)
    ev = Ev()
    gen = b.drag(ctx.labels, ev)
    next(gen)
    ev.type = "mouse_move"
    for x in (65, 70, 75):
        ev.position = (5, 60, x)
        next(gen)
    ev.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(gen)
    assert ctx.labels.data[5, 60, 75] == 1 and ctx.labels.data[6, 60, 75] == 0
    ctx.labels.undo()
    assert count(ctx) == 0


def test_outer_boundary_ignores_holes():
    from ctseg.tools.morphology import outer_boundary
    m = np.zeros((30, 30), bool); m[5:25, 5:25] = True; m[12:18, 12:18] = False   # square with a hole
    o = outer_boundary(m, 1)
    assert o.sum() == 19 * 4 and not o[12, 12] and not o[11, 11]
    out = outer_boundary(m, 1, outside=True)
    assert out.sum() == 20 * 4 and not out[5, 5]


# ── multi-Otsu, multi-seed, 3D view, STL ─────────────────────────────────────

def test_multi_otsu_slice_and_volume_with_protect(app):
    v, ctx = app
    o = ctx.panels["otsu"]
    o.classes.value = 3
    o()                                                     # this slice
    assert count(ctx) > 0 and count(ctx, 6) == 0
    assert {1, 2} <= set(np.unique(ctx.labels.data[5]))
    assert 1 in ctx.table.entries and 2 in ctx.table.entries      # classes got names and colours
    ctx.labels.undo()
    assert count(ctx) == 0
    ctx.labels.data[5, 60, 70] = 9                          # an existing label must survive protect
    o.scope.value = "whole volume"
    o()
    assert ctx.labels.data[5, 60, 70] == 9
    assert count(ctx, 0) > 0 and count(ctx, 11) > 0


def test_multi_seed_grow_two_materials(app):
    v, ctx = app
    ctx.labels.selected_label = 1
    v.cursor.position = (5, 60, 35)                         # inside the ring (bright)
    ctx.panels["seeds"].add()
    ctx.labels.selected_label = 2
    v.cursor.position = (5, 5, 5)                           # background corner (dark)
    ctx.panels["seeds"].add()
    assert len(ctx.seeds) == 2 and len(ctx.seeds_layer.data) == 2
    ctx.panels["seeds"].grow()
    d = ctx.labels.data[5]
    assert (d == 1).sum() > 2000 and (d == 2).sum() > 2000
    assert d[ctx.ring].mean() > 0.9 and (d[ctx.ring] == 1).mean() > 0.9     # ring is label 1
    assert d[5, 5] == 2 and d[110, 130] == 2                                 # outside the ring is label 2
    assert count(ctx, 6) == 0                                                # 2D mode: seed slice only
    ctx.labels.undo()
    assert count(ctx) == 0
    g = ctx.panels["seeds"].grow
    g.scope.value = "3D slab around seeds"
    g()
    assert count(ctx, 6) > 2000 and count(ctx, 11) == 0
    ctx.panels["seeds"].clear()
    assert ctx.seeds == []


def test_view_panel_3d_toggle_and_stl(app, tmp_path):
    v, ctx = app
    view = ctx.panels["view"]
    view.toggle()
    assert v.dims.ndisplay == 3 and ctx.labels.rendering == "iso_categorical"
    assert ctx.preview_layer.visible is False
    view.toggle()
    assert v.dims.ndisplay == 2 and ctx.preview_layer.visible is True
    view.export.path.value = tmp_path / "empty.stl"
    view.export()                                           # empty label: a message, not a crash
    assert "empty" in v.status and not (tmp_path / "empty.stl").exists()
    v.keys["Ctrl-G"](v)
    ctx.commit(0, np.where(ctx.volume[:3] > 100, 1, 0).astype(ctx.labels.data.dtype))
    view.export.path.value = tmp_path / "ring.stl"
    view.export()
    size = (tmp_path / "ring.stl").stat().st_size
    assert size > 84 and (size - 84) % 50 == 0
