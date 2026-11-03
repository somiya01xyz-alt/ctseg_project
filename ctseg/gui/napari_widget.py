"""
Interactive CT segmentation in napari.

napari's own Labels layer already provides the brush, eraser, bucket fill and a label
picker. This module adds what napari does not have, as tabs on the right:

    Select : threshold (live preview), multi-Otsu, multi-seed growing, flood fill, connected components
    Brush  : round / square, 2D / 3D brush and eraser (stamp or drag)
    Edit   : dilate / erode / open / close / fill, propagation, interpolation, thin structures
    ROI    : rectangle, ellipse, polygon and freehand ROIs, statistics, save / load
    Labels : label names and colours, save / load project, Dice against a reference folder
    View   : 2D / 3D switch, show only the active label, export a label as an STL mesh
    Help   : keyboard shortcuts

Run with:
    python -m ctseg.gui.napari_widget /path/to/Assembly_folder [downsample]

Every button writes through napari's own history, so Ctrl+Z / Ctrl+Shift+Z undo and
redo them. Ctrl+G grows a region from the cursor; Ctrl+K keeps the component under it.
"""
import sys

import numpy as np
from magicgui.widgets import Container

from ..fileio.files import load_tiff_stack
from ..tools.select import region_grow_auto
from .panels import (Ctx, brush_panel, components_panel, flood_panel, edit_panel, help_panel, label_panel,
                     multi_otsu_panel, roi_panel, seeds_panel, threshold_panel, view_panel)

# Region-grow settings chosen by examples/tune_region_grow.py on the real CT:
# Dice on Ceramic 0.54 -> 0.87 and on Metal 0.38 -> 0.94 (old defaults were k=2, radius=4).
# Lower k (e.g. 1.5) if the red leaks into neighbouring material.
GROW_K = 4.0
GROW_RADIUS = 8


def _make_narrow(widget, min_width=260):
    """Stop long text from forcing the dock panel wide.

    A QLabel without word-wrap reports its full single-line text width as its minimum
    width, so one long hint line makes the whole dock too wide to shrink again.
    """
    from qtpy.QtWidgets import QLabel, QSizePolicy
    native = widget.native
    for lab in native.findChildren(QLabel):
        lab.setWordWrap(True)
        lab.setMinimumWidth(0)
        pol = lab.sizePolicy()
        pol.setHorizontalPolicy(QSizePolicy.Ignored)   # width comes from the dock, not the text
        pol.setHeightForWidth(True)
        lab.setSizePolicy(pol)
    native.setMinimumWidth(min_width)
    return native


def _scrollable(widget):
    """Wrap a panel in a scroll area so its contents never force a minimum window size.

    Without this the Select tab's minimum height/width pushes the whole napari window
    taller than the screen. With it the panel scrolls instead.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QScrollArea
    native = _make_narrow(widget)
    native.setMinimumWidth(0)
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    area.setFrameShape(QScrollArea.NoFrame)
    area.setMinimumWidth(260)
    area.setWidget(native)
    return area


def build(viewer, volume, downsample=1):
    """Attach the CT, a Labels layer and all panels to `viewer`. Returns the Ctx."""
    image_layer = viewer.add_image(volume, name="CT volume")
    labels_layer = viewer.add_labels(np.zeros(volume.shape, dtype=np.uint8), name="labels")
    ctx = Ctx(viewer, volume, labels_layer, downsample)
    ctx.image_layer = image_layer

    def region_grow_at_cursor(viewer):
        """Ctrl+G: region-grow a segment starting at the current cursor position."""
        z, y, x = ctx.cursor_voxel()
        label = labels_layer.selected_label
        mask3d = region_grow_auto(volume, (z, y, x), k=GROW_K, radius=GROW_RADIUS)
        block = labels_layer.data[z:z + 1].copy()
        block[0][mask3d[z]] = label
        n = ctx.commit(z, block)
        ctx.say(f"region grow @ ({z},{y},{x}) label={label} voxels={mask3d[z].sum()} "
                f"(changed {n}; Ctrl+Z to undo)")

    viewer.bind_key("Ctrl-G")(region_grow_at_cursor)
    ctx.region_grow_at_cursor = region_grow_at_cursor

    select = components_panel(ctx)
    threshold = threshold_panel(ctx)
    flood = flood_panel(ctx)
    otsu = multi_otsu_panel(ctx)
    seeds = seeds_panel(ctx)
    select_tab = Container(widgets=[threshold, otsu, seeds, flood, select], labels=False)
    brush = brush_panel(ctx)
    edit = edit_panel(ctx)
    roi = roi_panel(ctx)
    labels = label_panel(ctx)
    labels.add_one("Ceramic")           # start with one label so the tool is usable at once
    view = view_panel(ctx)
    help_ = help_panel(ctx)

    ctx.panels = {"threshold": threshold, "components": select, "edit": edit, "flood": flood, "brush": brush,
                  "roi": roi, "labels": labels, "help": help_,
                  "otsu": otsu, "seeds": seeds, "view": view}
    for name, widget in [("Select", select_tab), ("Brush", brush), ("Edit", edit), ("ROI", roi),
                         ("Labels", labels), ("View", view), ("Help", help_)]:
        viewer.window.add_dock_widget(_scrollable(widget), area="right", name=name, tabify=True)
    return ctx


def launch(volume_folder, downsample=1):
    import napari

    volume = load_tiff_stack(volume_folder, downsample=downsample)
    viewer = napari.Viewer(title="ctseg — Interactive CT Segmentation")
    build(viewer, volume, downsample)
    print("[ctseg] scrollable panels v5 active, loaded from:", __file__)
    print(__doc__)
    napari.run()


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/Assembly"
    ds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    launch(folder, downsample=ds)
