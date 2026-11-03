"""
Unit tests — run with:  pytest -v

These use small made-up arrays (not your CT data) so they run in under a
second and don't depend on any files being present. Read them top to bottom;
each test is a worked example of how to call that tool.
"""
import numpy as np
import pytest

from ctseg.core.labelmap import LabelMap
from ctseg.core.label_table import LabelTable
from ctseg.tools.brush import stroke_mask, paint_stroke, erase_stroke
from ctseg.tools.roi import ROI
from ctseg.tools.select import (threshold_mask, flood_fill, region_grow,
                                 region_grow_auto, component_at,
                                 largest_component, remove_small)
from ctseg.tools.select import multi_otsu, multi_seed_grow
from ctseg.tools.mesh import mask_to_mesh, write_stl
from ctseg.tools.morphology import dilate, erode, opening, closing, apply_morphology
from ctseg.tools.propagate import interpolate_between_slices, propagate_slices
from ctseg.evaluate import dice, iou, per_slice_dice


# ---------------------------------------------------------------- LabelMap

def test_paint_and_erase():
    lm = LabelMap((5, 10, 10))
    lm.paint_slice(2, np.ones((10, 10), bool), label=1)
    assert (lm.data[2] == 1).all()
    assert (lm.data[0] == 0).all()          # other slices untouched

    lm.erase_slice(2, np.ones((10, 10), bool))
    assert (lm.data[2] == 0).all()


def test_protect_does_not_overwrite_other_labels():
    lm = LabelMap((1, 4, 4))
    lm.paint_slice(0, np.array([[1, 1, 0, 0]] * 4, bool), label=1)
    lm.paint_slice(0, np.ones((4, 4), bool), label=2, protect=True)
    # label-1 pixels must still be 1, the rest become 2
    assert lm.data[0, 0, 0] == 1
    assert lm.data[0, 0, 2] == 2


def test_undo_redo():
    lm = LabelMap((1, 4, 4))
    lm.paint_slice(0, np.ones((4, 4), bool), label=1)
    assert (lm.data == 1).all()
    lm.undo()
    assert (lm.data == 0).all()
    lm.redo()
    assert (lm.data == 1).all()
    assert lm.undo() is True
    assert lm.undo() is False   # nothing left to undo


def test_undo_stores_only_the_changed_box():
    lm = LabelMap((1, 100, 100))
    lm.paint_slice(0, np.zeros((100, 100), bool) | _square(10, 10, 5), label=1)
    box, old, new = lm._undo[-1]
    # the stored box should be small, not the whole 100x100 slice
    assert new.size < 100 * 100


def _square(y, x, r):
    m = np.zeros((100, 100), bool)
    m[y - r:y + r, x - r:x + r] = True
    return m


# -------------------------------------------------------------------- Brush

def test_brush_stroke_paints_a_line():
    lm = LabelMap((1, 50, 50))
    points = [(0, 10, 10), (0, 10, 40)]   # a horizontal line
    paint_stroke(lm, points, radius=2, label=1, mode="2d")
    assert lm.data[0, 10, 10] == 1
    assert lm.data[0, 10, 25] == 1        # interpolation filled the middle
    assert lm.data[0, 10, 40] == 1
    assert lm.data[0, 40, 40] == 0        # far away: untouched


def test_eraser_removes_only_painted_voxels():
    lm = LabelMap((1, 50, 50))
    paint_stroke(lm, [(0, 25, 25)], radius=10, label=1)
    assert lm.data[0, 25, 25] == 1
    erase_stroke(lm, [(0, 25, 25)], radius=3)
    assert lm.data[0, 25, 25] == 0
    assert lm.data[0, 25, 33] == 1        # outside eraser radius, still painted


# ---------------------------------------------------------------------- ROI

def test_roi_rectangle_mask_and_stats():
    roi = ROI(kind="rectangle", z=0, points=[(2, 2), (5, 5)])
    mask = roi.to_mask((10, 10))
    assert mask.sum() == 4 * 4            # inclusive 2..5 both axes
    image = np.full((10, 10), 7.0)
    stats = roi.stats(image)
    assert stats["mean"] == 7.0
    assert stats["area"] == mask.sum()


def test_roi_polygon_mask():
    roi = ROI(kind="polygon", z=0, points=[(1, 1), (1, 5), (5, 5), (5, 1)])
    mask = roi.to_mask((10, 10))
    assert mask[3, 3]          # center of the square is inside
    assert not mask[0, 0]      # corner far outside


# ----------------------------------------------------------------- Selection

def test_threshold_mask():
    image = np.arange(25).reshape(5, 5)
    m = threshold_mask(image, low=5, high=10)
    assert m.sum() == 6        # values 5,6,7,8,9,10


def test_flood_fill_stops_at_value_change():
    image = np.zeros((10, 10))
    image[3:7, 3:7] = 100
    m = flood_fill(image, seed=(5, 5), tolerance=1)
    assert m[5, 5] and m[3, 3]
    assert not m[0, 0]


def test_region_grow_respects_connectivity():
    image = np.zeros((10, 10))
    image[2, 2] = 50           # a lone pixel, not touching the big blob
    image[5:8, 5:8] = 50
    m = region_grow(image, seed=(6, 6), low=40, high=60)
    assert m[6, 6]
    assert not m[2, 2]         # separate component, must be excluded


def test_region_grow_auto_finds_a_blob_without_knowing_its_range():
    # The sampling neighborhood (radius=3 -> 7x7) must fit INSIDE the blob,
    # or it mixes foreground+background and the mean/std estimate breaks.
    # This matters for real use too: don't seed region_grow_auto too close
    # to a material's edge, or shrink `radius`.
    image = np.full((30, 30), 10.0)
    image[10:20, 10:20] = 200.0   # 10x10 blob, seed well inside it
    m = region_grow_auto(image, seed=(15, 15), radius=3)
    assert m[15, 15]
    assert m.sum() == 100          # exactly the 10x10 blob, no leakage


def test_connected_components():
    binary = np.zeros((10, 10), bool)
    binary[1:3, 1:3] = True     # 4 px
    binary[6:10, 6:10] = True   # 16 px, the bigger blob
    assert component_at(binary, (1, 1)).sum() == 4
    assert largest_component(binary).sum() == 16


def test_remove_small_objects():
    binary = np.zeros((20, 20), bool)
    binary[0:2, 0:2] = True      # 4 px -> should be removed
    binary[10:15, 10:15] = True  # 25 px -> should survive
    cleaned = remove_small(binary, min_size=10)
    assert cleaned.sum() == 25


# --------------------------------------------------------------- Morphology

def test_dilate_grows_and_erode_shrinks():
    m = np.zeros((20, 20), bool)
    m[10, 10] = True
    grown = dilate(m, radius=2)
    assert grown.sum() > 1
    shrunk = erode(grown, radius=2)
    assert shrunk.sum() <= grown.sum()


def test_opening_removes_a_thin_speck():
    m = np.zeros((20, 20), bool)
    m[5, 5] = True                       # isolated 1px speck
    m[10:15, 10:15] = True               # solid blob
    cleaned = opening(m, radius=1)
    assert not cleaned[5, 5]
    assert cleaned[12, 12]


def test_apply_morphology_on_labelmap():
    lm = LabelMap((1, 20, 20))
    mask = np.zeros((20, 20), bool); mask[9, 9] = True
    lm.paint_slice(0, mask, label=1)
    apply_morphology(lm, label=1, op="dilate", radius=2, z=0)
    assert lm.get_mask(1, z=0).sum() > 1


def test_apply_morphology_rejects_unknown_op():
    lm = LabelMap((1, 10, 10))
    with pytest.raises(ValueError):
        apply_morphology(lm, label=1, op="not_a_real_op")


# --------------------------------------------------------------- Propagate

def test_interpolate_between_slices_fills_the_middle():
    lm = LabelMap((5, 20, 20))
    disk_small = np.zeros((20, 20), bool); disk_small[8:12, 8:12] = True
    lm.paint_slice(0, disk_small, label=1)
    lm.paint_slice(4, disk_small, label=1)
    interpolate_between_slices(lm, label=1, z0=0, z1=4)
    for z in (1, 2, 3):
        assert lm.get_mask(1, z=z).sum() > 0


def test_interpolate_handles_swapped_z0_z1():
    lm = LabelMap((5, 20, 20))
    disk_small = np.zeros((20, 20), bool); disk_small[8:12, 8:12] = True
    lm.paint_slice(0, disk_small, label=1)
    lm.paint_slice(4, disk_small, label=1)
    assert interpolate_between_slices(lm, label=1, z0=4, z1=0) is True  # swapped, still works


def test_propagate_slices_tracks_a_bright_blob():
    # Blob must be bigger than propagate_slices' default min_size=20,
    # or remove_small() correctly deletes it as noise on the very first step.
    image = np.full((5, 30, 30), 10.0)
    for z in range(5):
        image[z, 10:20, 10:20] = 200.0   # 10x10 = 100px blob, same on every slice
    lm = LabelMap((5, 30, 30))
    lm.paint_slice(0, image[0] > 100, label=1)
    n = propagate_slices(lm, image, label=1, z_start=0, z_end=4)
    assert n == 4
    assert lm.get_mask(1, z=4).sum() > 0


# --------------------------------------------------------------- Evaluate

def test_dice_identical_is_one():
    a = np.zeros((10, 10), bool); a[2:5, 2:5] = True
    assert dice(a, a) == 1.0
    assert iou(a, a) == 1.0


def test_dice_no_overlap_is_zero():
    a = np.zeros((10, 10), bool); a[0:2, 0:2] = True
    b = np.zeros((10, 10), bool); b[8:10, 8:10] = True
    assert dice(a, b) == 0.0


def test_per_slice_dice_shape():
    a = np.zeros((3, 10, 10), bool)
    b = np.zeros((3, 10, 10), bool)
    a[0, 0:5, 0:5] = True
    b[0, 0:5, 0:5] = True   # slice 0 matches
    scores = per_slice_dice(a, b)
    assert scores.shape == (3,)
    assert scores[0] == 1.0
    assert scores[1] == 1.0   # both empty -> dice() defines this as 1.0


# ------------------------------------------------------------- Label table

def test_label_table_round_trip():
    t = LabelTable()
    cid = t.add("Ceramic")
    d = t.to_dict()
    t2 = LabelTable.from_dict(d)
    assert t2.entries[cid]["name"] == "Ceramic"


# ---------------------------------------------------------- multi-Otsu, seeds, mesh

def _three_material_volume():
    rng = np.random.default_rng(1)
    v = np.full((16, 40, 60), 20.0)
    v[:, :, 20:40] = 100
    v[:, :, 40:] = 200
    return (v + rng.normal(0, 4, v.shape)).clip(0, 255).astype(np.uint8)


def test_multi_otsu_separates_three_materials():
    v = _three_material_volume()
    lab, th = multi_otsu(v, classes=3)
    assert lab.dtype == np.uint8 and len(th) == 2 and 25 < th[0] < 95 and 105 < th[1] < 195
    assert (lab[:, :, :20] == 0).mean() > 0.95 and (lab[:, :, 20:40] == 1).mean() > 0.9
    assert (lab[:, :, 40:] == 2).mean() > 0.95


def test_multi_otsu_reuses_thresholds_for_one_slice():
    v = _three_material_volume()
    lab, th = multi_otsu(v, classes=3)
    one, th2 = multi_otsu(v[3], thresholds=th)
    assert np.array_equal(one, lab[3]) and np.array_equal(th, th2)


def test_multi_seed_grow_each_seed_gets_its_label_and_overlaps_resolve():
    img = np.full((40, 60), 20.0)
    img[:, 30:] = 100
    out = multi_seed_grow(img, [(1, (20, 10)), (2, (20, 50))], k=4, radius=4)
    assert (out[:, :30] == 1).all() and (out[:, 30:] == 2).all()
    two_same = multi_seed_grow(img, [(7, (20, 10)), (7, (5, 5))], k=4, radius=4)
    assert set(np.unique(two_same)) == {0, 7} or set(np.unique(two_same)) == {7}


def test_mesh_of_a_cube_is_closed_and_in_voxel_units(tmp_path):
    m = np.zeros((10, 10, 10), bool)
    m[2:8, 2:8, 2:8] = True
    verts, faces = mask_to_mesh(m)
    assert np.allclose(verts.min(0), 1.5) and np.allclose(verts.max(0), 7.5)
    v2, _ = mask_to_mesh(m, spacing=(2, 2, 2))
    assert np.allclose(v2.max(0), 15.0)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()                              # every edge shared by two triangles = closed
    edge = np.zeros((6, 6, 6), bool); edge[:3] = True       # touching the volume edge still closes
    ev, ef = mask_to_mesh(edge)
    e = np.sort(np.concatenate([ef[:, [0, 1]], ef[:, [1, 2]], ef[:, [2, 0]]]), axis=1)
    assert (np.unique(e, axis=0, return_counts=True)[1] == 2).all()
    n = write_stl(tmp_path / "c.stl", verts, faces)
    assert n == len(faces) and (tmp_path / "c.stl").stat().st_size == 84 + 50 * n


def test_mesh_rejects_empty_label():
    with pytest.raises(ValueError):
        mask_to_mesh(np.zeros((5, 5, 5), bool))
