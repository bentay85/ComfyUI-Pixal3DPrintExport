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


def _reduce_colors(colors: np.ndarray, count: int) -> np.ndarray:
    """Snap sRGB vertex colors to a palette of `count` colors (deterministic k-means)."""
    n = colors.shape[0]
    count = int(count)
    if count <= 0 or count >= n:
        return colors
    colors = colors.astype(np.float32)
    first = int(np.argmax(((colors - colors.mean(0)) ** 2).sum(1)))
    centers = [colors[first]]
    dist = ((colors - centers[0]) ** 2).sum(1)
    for _ in range(count - 1):
        idx = int(np.argmax(dist))
        centers.append(colors[idx])
        dist = np.minimum(dist, ((colors - colors[idx]) ** 2).sum(1))
    centers = np.stack(centers)
    for _ in range(32):
        cost = (colors ** 2).sum(1)[:, None] + (centers ** 2).sum(1)[None, :] - 2.0 * colors @ centers.T
        label = cost.argmin(1)
        updated = np.stack([colors[label == k].mean(0) if (label == k).any() else centers[k] for k in range(count)])
        if np.allclose(updated, centers):
            centers = updated
            break
        centers = updated
    cost = (colors ** 2).sum(1)[:, None] + (centers ** 2).sum(1)[None, :] - 2.0 * colors @ centers.T
    return centers[cost.argmin(1)]


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
            c = _reduce_colors(c, color_count)
        for i in range(v.shape[0]):
            lines.append(f"v {v[i, 0]:.6f} {v[i, 1]:.6f} {v[i, 2]:.6f} {c[i, 0]:.6f} {c[i, 1]:.6f} {c[i, 2]:.6f}")
    else:
        for i in range(v.shape[0]):
            lines.append(f"v {v[i, 0]:.6f} {v[i, 1]:.6f} {v[i, 2]:.6f}")
    for face in f:
        lines.append(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}")
    return ("\n".join(lines) + "\n").encode("utf-8")
