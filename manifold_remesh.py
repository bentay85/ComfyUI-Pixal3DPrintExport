"""UDF Manifold Dual Contouring.

ComfyUI's `remesh_narrow_band_dc` only enables Manifold DC when `sign_mode="sdf"`,
yet MDC is sign-agnostic. UDF is the mode that stays robust on messy /
self-intersecting meshes, so we drive the same narrow-band -> UDF -> MDC sequence
from here instead of changing core.
"""

import torch

from comfy_extras.mesh3d.postprocess.remesh import (
    _build_narrow_band_voxels,
    _dual_contour_manifold,
    _filter_components,
    _udf_exact,
)

_CORNER_OFFSETS = (
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
)


def remesh_manifold_udf(vertices: torch.Tensor, faces: torch.Tensor, resolution: int, band: float = 1.0):
    """Re-extract a watertight, 2-manifold surface; returns (vertices, faces).

    Same narrow-band UDF / Dual Contouring pipeline as core, but always taking
    the Manifold DC branch so multi-sheet layer pinches are split instead of
    left as 4-face edges.
    """
    device = vertices.device
    center = 0.5 * (vertices.max(dim=0)[0] + vertices.min(dim=0)[0])
    scale = float((vertices.max(dim=0)[0] - vertices.min(dim=0)[0]).max().item()) * 1.1
    eps = band * scale / resolution

    voxel_coords, tree = _build_narrow_band_voxels(vertices, faces, center, scale, resolution, eps)
    if voxel_coords.numel() == 0:
        return (torch.empty((0, 3), dtype=vertices.dtype, device=device),
                torch.empty((0, 3), dtype=faces.dtype, device=device))

    corner_offsets = torch.tensor(_CORNER_OFFSETS, dtype=torch.long, device=device)
    corners = (voxel_coords.unsqueeze(1) + corner_offsets.unsqueeze(0)).reshape(-1, 3)
    r1 = resolution + 1
    corner_keys = (corners[:, 0] * r1 + corners[:, 1]) * r1 + corners[:, 2]
    unique_corner_keys, corner_inv = torch.unique(corner_keys, return_inverse=True)
    unique_corners = torch.zeros((unique_corner_keys.shape[0], 3), dtype=torch.long, device=device)
    unique_corners[corner_inv] = corners

    tri_verts = vertices[faces.long()]
    corner_world = (unique_corners.float() / resolution - 0.5) * scale + center.unsqueeze(0)
    udf, _, closest_tri = _udf_exact(corner_world, tri_verts, tree=tree)
    corner_valid = closest_tri >= 0

    dual_verts, new_faces = _dual_contour_manifold(
        voxel_coords, udf - eps, unique_corner_keys,
        resolution, scale, center, corner_valid=corner_valid)

    if new_faces.numel() > 0:
        new_faces = _filter_components(dual_verts, new_faces,
                                       min_fraction=0.01, drop_inverted=True, drop_enclosed=True)
    if dual_verts.numel() > 0 and new_faces.numel() > 0:
        used = torch.zeros(dual_verts.shape[0], dtype=torch.bool, device=device)
        used[new_faces[:, 0]] = True
        used[new_faces[:, 1]] = True
        used[new_faces[:, 2]] = True
        dual_verts = dual_verts[used]
        new_faces = (used.long().cumsum(0) - 1)[new_faces.long()]

    return dual_verts.to(vertices.dtype), new_faces.to(faces.dtype)
