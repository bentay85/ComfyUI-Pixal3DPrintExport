import os

import numpy as np
import torch

from comfy_extras.mesh3d.fileio import gltf_read, mesh_file_read

from comfy_api.latest import Types


def file3d_to_tensors(mesh: Types.File3D) -> tuple[torch.Tensor, torch.Tensor]:
    data = mesh.get_bytes()
    fmt = (mesh.format or "").lower()
    if fmt in ("glb", "gltf"):
        base_dir = os.path.dirname(mesh.get_source()) if mesh.is_disk_backed else None
        _, _, prims = gltf_read.load_gltf(data, base_dir, lambda k, m: None)
        if not prims:
            raise ValueError("no triangle geometry found in the glTF scene")
        positions = np.concatenate([p["positions"] for p in prims], axis=0)
        offset = 0
        all_faces = []
        for p in prims:
            all_faces.append(p["faces"] + offset)
            offset += p["positions"].shape[0]
        faces = np.concatenate(all_faces, axis=0)
    elif fmt == "obj":
        prim = mesh_file_read.load_obj(data)
        positions = prim["positions"]
        faces = prim["faces"]
    elif fmt == "stl":
        prim = mesh_file_read.load_stl(data)
        positions = prim["positions"]
        faces = prim["faces"]
    else:
        raise ValueError(f"unsupported format {fmt!r}")
    return torch.from_numpy(positions.astype(np.float32)), torch.from_numpy(faces.astype(np.int64))


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.00313, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


# sRGB <-> CIELAB (D65). Quantizing in Lab keeps the palette perceptually even instead of
# spending entries on shading steps that are barely visible but split a flat region in RGB.
_D65 = np.array([0.95047, 1.0, 1.08883])
_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_XYZ_TO_RGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])
_LAB_EPS = 216.0 / 24389.0
_LAB_KAPPA = 24389.0 / 27.0


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ _RGB_TO_XYZ.T) / _D65
    f = np.where(xyz > _LAB_EPS, np.cbrt(xyz), (_LAB_KAPPA * xyz + 16.0) / 116.0)
    return np.stack([
        116.0 * f[..., 1] - 16.0,
        500.0 * (f[..., 0] - f[..., 1]),
        200.0 * (f[..., 1] - f[..., 2]),
    ], axis=-1)


def _lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16.0) / 116.0
    f = np.stack([fy + lab[..., 1] / 500.0, fy, fy - lab[..., 2] / 200.0], axis=-1)
    f3 = f ** 3
    xyz = np.where(f3 > _LAB_EPS, f3, (116.0 * f - 16.0) / _LAB_KAPPA) * _D65
    lin = np.clip(xyz @ _XYZ_TO_RGB.T, 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0)


def _vertex_area_weights(vertices: np.ndarray, faces: np.ndarray, n: int) -> np.ndarray:
    """Weight each vertex by the surface area it owns, so large flat regions drive the
    palette instead of dense noisy patches with thousands of tiny triangles."""
    weights = np.zeros(n, dtype=np.float64)
    if faces.shape[0]:
        a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        for k in range(3):
            weights += np.bincount(faces[:, k], weights=area / 3.0, minlength=n)
    if weights.sum() <= 0.0:
        weights[:] = 1.0
    return weights


def _kmeans_lab(points: np.ndarray, count: int, weights: np.ndarray) -> np.ndarray:
    """Area-weighted farthest-point seeding plus weighted Lloyd iterations in Lab space."""
    centers = [points[int(np.argmax(weights))]]
    dist = ((points - centers[0]) ** 2).sum(1)
    for _ in range(count - 1):
        idx = int(np.argmax(dist * weights))
        centers.append(points[idx])
        dist = np.minimum(dist, ((points - points[idx]) ** 2).sum(1))
    centers = np.stack(centers)
    for _ in range(32):
        cost = (points ** 2).sum(1)[:, None] + (centers ** 2).sum(1)[None, :] - 2.0 * points @ centers.T
        label = cost.argmin(1)
        updated = []
        for k in range(count):
            mask = label == k
            if mask.any():
                wk = weights[mask]
                updated.append((points[mask] * wk[:, None]).sum(0) / wk.sum())
            else:
                updated.append(centers[k])
        updated = np.stack(updated)
        if np.allclose(updated, centers):
            centers = updated
            break
        centers = updated
    return centers


def _reduce_colors(colors: np.ndarray, vertices: np.ndarray, faces: np.ndarray, count: int) -> np.ndarray:
    """Snap sRGB vertex colors to a palette of `count` colors (deterministic k-means in CIELAB,
    weighted by vertex area)."""
    n = colors.shape[0]
    count = int(count)
    if count <= 0 or count >= n:
        return colors
    lab = _srgb_to_lab(colors.astype(np.float64))
    centers = _kmeans_lab(lab, count, _vertex_area_weights(vertices, faces, n))
    cost = (lab ** 2).sum(1)[:, None] + (centers ** 2).sum(1)[None, :] - 2.0 * lab @ centers.T
    return _lab_to_srgb(centers[cost.argmin(1)]).astype(np.float32)


def mesh_to_obj_bytes(vertices: torch.Tensor, faces: torch.Tensor, vertex_colors: torch.Tensor | None = None,
                      color_count: int = 0) -> bytes:
    v = vertices.cpu().numpy().astype(np.float32)
    f = faces.cpu().numpy().astype(np.int64)
    first = None
    if v.shape[0] > 0:
        extent = float(v.max() - v.min())
        tol = max(extent * 1e-6, 1e-6)
        key = np.round(v / tol).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        v = v[first]
        f = inv[f]
        if f.size:
            f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    lines = []
    if vertex_colors is not None:
        c = np.clip(_linear_to_srgb(np.clip(vertex_colors.cpu().numpy().astype(np.float32)[:, :3], 0.0, 1.0)), 0.0, 1.0)
        if first is not None:
            c = c[first]
        if color_count > 0:
            c = _reduce_colors(c, v, f, color_count)
        for i in range(v.shape[0]):
            lines.append(f"v {v[i, 0]:.6f} {v[i, 1]:.6f} {v[i, 2]:.6f} {c[i, 0]:.6f} {c[i, 1]:.6f} {c[i, 2]:.6f}")
    else:
        for i in range(v.shape[0]):
            lines.append(f"v {v[i, 0]:.6f} {v[i, 1]:.6f} {v[i, 2]:.6f}")
    for face in f:
        lines.append(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}")
    return ("\n".join(lines) + "\n").encode("utf-8")
