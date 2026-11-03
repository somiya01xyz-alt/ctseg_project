"""
2-D Region-of-Interest (ROI) tools.

An ROI stores its *shape* (a few points), not the pixels.
`roi.to_mask(shape2d)` rasterises it when you need pixels.
Points are (y, x) in image coordinates.

ROIs are saved to / loaded from JSON so you can edit them later.
"""
import json
from dataclasses import dataclass, field, asdict
import numpy as np
from skimage.draw import ellipse, polygon


@dataclass
class ROI:
    kind: str          # "rectangle" | "ellipse" | "polygon" | "freehand"
    z: int             # which CT slice
    points: list = field(default_factory=list)
    # rectangle / ellipse: [top-left (y,x), bottom-right (y,x)]
    # polygon / freehand : list of (y,x) vertices
    name: str = ""

    # ── rasterise to a 2-D bool mask ─────────────────────────────────────────
    def to_mask(self, shape2d: tuple) -> np.ndarray:
        m = np.zeros(shape2d, bool)
        p = np.asarray(self.points, float)
        if self.kind in ("rectangle", "ellipse"):
            y0, x0 = p.min(axis=0)
            y1, x1 = p.max(axis=0)
            if self.kind == "rectangle":
                m[int(round(y0)):int(round(y1)) + 1,
                  int(round(x0)):int(round(x1)) + 1] = True
            else:
                cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
                ry, rx = max((y1 - y0) / 2, 0.5), max((x1 - x0) / 2, 0.5)
                rr, cc = ellipse(cy, cx, ry, rx, shape2d)
                m[rr, cc] = True
        elif self.kind in ("polygon", "freehand"):
            rr, cc = polygon(p[:, 0], p[:, 1], shape2d)
            m[rr, cc] = True
        else:
            raise ValueError(f"Unknown ROI kind: {self.kind!r}")
        return m

    # ── intensity stats inside the ROI ───────────────────────────────────────
    def stats(self, image2d: np.ndarray) -> dict:
        mask = self.to_mask(image2d.shape)
        v = image2d[mask]
        if v.size == 0:
            return {"area": 0}
        return {
            "area": int(v.size),
            "mean": float(v.mean()),
            "std":  float(v.std()),
            "min":  float(v.min()),
            "max":  float(v.max()),
        }


# ── save / load ───────────────────────────────────────────────────────────────

def save_rois(rois: list, path: str):
    with open(path, "w") as f:
        json.dump([asdict(r) for r in rois], f, indent=2)


def load_rois(path: str) -> list:
    with open(path) as f:
        return [ROI(**d) for d in json.load(f)]
