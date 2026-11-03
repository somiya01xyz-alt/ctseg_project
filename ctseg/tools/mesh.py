"""
Surface meshes from a label: marching cubes -> vertices and triangles -> binary STL file.
No napari here, only numpy and scikit-image.
"""
import numpy as np


def mask_to_mesh(mask: np.ndarray, spacing=(1.0, 1.0, 1.0), step_size: int = 1):
    """
    Triangle mesh of the surface of a 3D bool mask.
    spacing   : size of one voxel along (z, y, x). Use the downsample factor to keep the mesh in
                full-resolution voxel units, e.g. (2, 2, 2) for a volume loaded with downsample=2.
    step_size : marching-cubes step; 2 or 3 makes a coarser, much smaller mesh of a big label.
    Returns (vertices (n,3) in z, y, x order, faces (m,3)).
    Only the box around the label is processed, and it is padded by one voxel so the mesh is
    closed where the label touches the edge of the volume.
    """
    from skimage.measure import marching_cubes
    mask = np.asarray(mask, bool)
    if mask.ndim != 3 or not mask.any():
        raise ValueError("the label is empty (or the mask is not 3D): nothing to export")
    idx = np.nonzero(mask.any(axis=(1, 2)))[0], np.nonzero(mask.any(axis=(0, 2)))[0], \
        np.nonzero(mask.any(axis=(0, 1)))[0]
    lo = [int(i[0]) for i in idx]
    box = tuple(slice(l, int(i[-1]) + 1) for l, i in zip(lo, idx))
    sub = np.pad(mask[box], 1).astype(np.float32)
    verts, faces, _, _ = marching_cubes(sub, level=0.5, spacing=tuple(float(s) for s in spacing),
                                        step_size=max(int(step_size), 1))
    verts = verts + (np.array(lo, float) - 1.0) * np.array(spacing, float)
    return verts, faces


def write_stl(path, verts: np.ndarray, faces: np.ndarray) -> int:
    """Write a binary STL (x, y, z order, as STL viewers expect). Returns the triangle count."""
    v = np.asarray(verts, np.float32)[:, ::-1]          # (z, y, x) -> (x, y, z)
    tri = v[np.asarray(faces)]                           # (m, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, length, out=np.zeros_like(n), where=length > 0)
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as f:
        f.write(b"ctseg binary STL".ljust(80, b" "))
        f.write(np.uint32(len(tri)).tobytes())
        f.write(rec.tobytes())
    return len(tri)


def export_label_stl(labels: np.ndarray, label: int, path, spacing=(1.0, 1.0, 1.0),
                     step_size: int = 1) -> int:
    """Convenience: mesh of `labels == label` written to `path`. Returns the triangle count."""
    verts, faces = mask_to_mesh(labels == label, spacing, step_size)
    return write_stl(path, verts, faces)
