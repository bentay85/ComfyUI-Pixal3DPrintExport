import logging

import torch

try:
    from comfy_extras.mesh3d.postprocess.qem_decimate import (
        clean_mesh,
        qem_decimate_simplify,
        _repair_nonmanifold_edges,
    )
    from comfy_extras.nodes_mesh_postprocess import fill_holes_v2_fn
except ImportError as e:
    raise ImportError(
        "ComfyUI-Pixal3DPrintExport needs a ComfyUI version that provides "
        "mesh3d.postprocess.qem_decimate and nodes_mesh_postprocess.fill_holes_v2_fn. "
        "Update ComfyUI and restart."
    ) from e

from comfy_api.latest import IO, Types
from typing_extensions import override

from .manifold_remesh import remesh_manifold_udf
from .mesh_file import file3d_to_tensors


def _has_nonmanifold_edges(faces: torch.Tensor) -> bool:
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0).long()
    edges, _ = torch.sort(edges, dim=1)
    packed = edges[:, 0] * (int(faces.max()) + 1) + edges[:, 1]
    _, counts = torch.unique(packed, return_counts=True)
    return bool((counts > 2).any())


def _has_boundary_edges(faces: torch.Tensor) -> bool:
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0).long()
    edges, _ = torch.sort(edges, dim=1)
    packed = edges[:, 0] * (int(faces.max()) + 1) + edges[:, 1]
    _, counts = torch.unique(packed, return_counts=True)
    return bool((counts == 1).any())


def _drop_bad_faces(faces: torch.Tensor) -> torch.Tensor:
    """Drop degenerate faces and collapse duplicated faces (a pinched Dual Contouring
    shell can emit the same triangle on both sheets, which the fill stage cannot fix)."""
    faces = faces.long()
    canonical = torch.sort(faces, dim=1).values
    ok = (canonical[:, 0] != canonical[:, 1]) & (canonical[:, 1] != canonical[:, 2])
    faces, canonical = faces[ok], canonical[ok]
    unique, inverse = torch.unique(canonical, dim=0, return_inverse=True)
    first = torch.full((unique.shape[0],), faces.shape[0], dtype=torch.long, device=faces.device)
    first.scatter_reduce_(0, inverse, torch.arange(faces.shape[0], device=faces.device), reduce="amin")
    keep = torch.zeros(faces.shape[0], dtype=torch.bool, device=faces.device)
    keep[first] = True
    return faces[keep]


def _compact(vertices: torch.Tensor, faces: torch.Tensor):
    used = torch.zeros(vertices.shape[0], dtype=torch.bool, device=vertices.device)
    used[faces.reshape(-1)] = True
    remap = used.long().cumsum(0) - 1
    return vertices[used], remap[faces.long()]


def _weld(vertices: torch.Tensor, faces: torch.Tensor, rel_tol: float = 1e-6):
    """Snap vertices onto the same L_inf grid SaveOBJ uses on export, so the saved mesh is
    already at that weld's fixed point and the export cannot re-fuse distinct vertices."""
    extent = float(vertices.max() - vertices.min())
    tol = max(extent * rel_tol, 1e-6)
    keys, inverse = torch.unique(torch.round(vertices / tol).long(), dim=0, return_inverse=True)
    return keys.to(vertices.dtype) * tol, inverse[faces]


def _separate_pinches(vertices: torch.Tensor, faces: torch.Tensor, src: torch.Tensor, delta: float) -> torch.Tensor:
    """Nudge apart the coincident corners that manifold-edge repair duplicated. The pinched
    sheets then stop sharing a position, so a downstream position-weld cannot re-fuse them."""
    row = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0).long()
    centroid = torch.zeros_like(vertices)
    count = torch.zeros(vertices.shape[0], 1, device=vertices.device, dtype=vertices.dtype)
    ones = torch.ones(row.shape[0], 1, device=vertices.device, dtype=vertices.dtype)
    for a, b in ((0, 1), (1, 0)):
        centroid.index_add_(0, row[:, a], vertices[row[:, b]])
        count.index_add_(0, row[:, a], ones)
    direction = centroid / count.clamp_min(1) - vertices
    norm = direction.norm(dim=1, keepdim=True)
    direction = torch.where(norm > 1e-9, direction / norm.clamp_min(1e-9), torch.zeros_like(direction))
    duplicate = (torch.bincount(src)[src] > 1).unsqueeze(1)
    return vertices + direction * (delta * duplicate)


def _close_holes(vertices: torch.Tensor, faces: torch.Tensor):
    """Cap every open boundary loop with a centroid fan. Each loop gets a fresh centroid, so
    unlike ComfyUI's vertex-fan filler this can never reuse an existing edge - it only closes
    the cracks Dual Contouring leaves, without emitting non-manifold or duplicate faces."""
    device = vertices.device
    edges = torch.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=1).reshape(-1, 2).long()
    v_max = int(faces.max()) + 1
    directed = edges[:, 0] * v_max + edges[:, 1]
    reverse = edges[:, 1] * v_max + edges[:, 0]
    sorted_directed, _ = torch.sort(directed)
    slot = torch.searchsorted(sorted_directed, reverse).clamp_max(sorted_directed.shape[0] - 1)
    boundary = edges[sorted_directed[slot] != reverse]
    successor = {}
    for a, b in boundary.cpu().numpy().tolist():
        successor[a] = b
    loops, visited = [], set()
    for start in successor:
        if start in visited:
            continue
        loop, node = [], start
        while node in successor and node not in visited:
            visited.add(node)
            loop.append(node)
            node = successor[node]
        if len(loop) >= 3 and node == start:
            loops.append(loop)
    if not loops:
        return vertices, faces
    centroids = torch.stack([vertices[torch.tensor(loop, device=device)].mean(dim=0) for loop in loops])
    fans = torch.tensor(
        [[b, a, len(vertices) + i] for i, loop in enumerate(loops) for a, b in zip(loop, loop[1:] + loop[:1])],
        dtype=faces.dtype, device=device)
    return torch.cat([vertices, centroids.to(vertices.dtype)], dim=0), torch.cat([faces, fans], dim=0)


def _resolve_topology(vertices: torch.Tensor, faces: torch.Tensor):
    """Iterate close + repair until the shell is 2-manifold and watertight. Closed loops are
    capped by `_close_holes`; any open chain left over (e.g. from a seam explosion) is capped
    by ComfyUI's chain filler without welding, so repair-duplicated vertices stay separated."""
    for _ in range(6):
        faces = _drop_bad_faces(faces)
        if faces.shape[0] == 0:
            break
        if _has_boundary_edges(faces):
            vertices, faces = _close_holes(vertices, faces)
            faces = _drop_bad_faces(faces)
            if _has_boundary_edges(faces):
                vertices, faces, _ = fill_holes_v2_fn(
                    vertices, faces, max_perimeter=1e30, max_verts=10_000_000,
                    fill_chains=True, weld_epsilon_rel=0.0)
                faces = _drop_bad_faces(faces)
        if not _has_nonmanifold_edges(faces) and not _has_boundary_edges(faces):
            break
        if _has_nonmanifold_edges(faces):
            diag = float(torch.norm(vertices.max(dim=0).values - vertices.min(dim=0).values))
            vertices, faces, src = _repair_nonmanifold_edges(vertices, faces)
            vertices = _separate_pinches(vertices, faces, src, diag * 1e-4)
    return vertices, faces


class PrepareSTLMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PrepareSTLMesh",
            display_name="Prepare Mesh (3D Print)",
            search_aliases=["watertight", "fill holes", "remesh", "decimate", "clean mesh", "3d print"],
            category="Pixal3DPrintExport",
            description="Clean, remesh (watertight) and decimate a mesh for 3D printing. Outputs a MESH.",
            inputs=[
                IO.MultiType.Input(
                    "mesh",
                    types=[
                        IO.Mesh,
                        IO.File3DSTL,
                        IO.File3DGLB,
                        IO.File3DGLTF,
                        IO.File3DOBJ,
                        IO.File3DAny,
                    ],
                    tooltip="Mesh or 3D file to clean up.",
                ),
                IO.Int.Input(
                    "remesh_resolution",
                    default=256,
                    min=0,
                    max=1024,
                    tooltip="Dual Contouring voxel resolution. 0 disables remeshing.",
                ),
                IO.Int.Input(
                    "target_faces",
                    default=100000,
                    min=0,
                    max=10_000_000,
                    tooltip="Decimate to this face count. 0 disables decimation.",
                ),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            outputs=[
                IO.Mesh.Output("mesh", tooltip="Cleaned, watertight mesh ready for texturing/printing."),
            ],
        )

    @classmethod
    def execute(cls, mesh: Types.MESH | Types.File3D, remesh_resolution: int, target_faces: int) -> IO.NodeOutput:
        if isinstance(mesh, Types.File3D):
            items = [file3d_to_tensors(mesh)]
        else:
            items = []
            for i in range(mesh.vertices.shape[0]):
                v = mesh.vertices[i]
                f = mesh.faces[i]
                if mesh.vertex_counts is not None:
                    v = v[:mesh.vertex_counts[i]]
                if mesh.face_counts is not None:
                    f = f[:mesh.face_counts[i]]
                items.append((v, f))

        out_v, out_f, v_counts, f_counts = [], [], [], []
        for i, (v, f) in enumerate(items):
            if f.shape[0] == 0:
                logging.warning(f"PrepareSTLMesh: skipping empty mesh at batch index {i}")
                continue
            device = v.device
            v, f, _, _, _ = clean_mesh(v, f)
            if remesh_resolution > 0 and f.shape[0] > 0:
                # Manifold DC keeps the shell 2-manifold, at the cost of sub-voxel
                # cracks where the two UDF layers pinch (closed below).
                v, f = remesh_manifold_udf(v, f, remesh_resolution)
                f = _drop_bad_faces(f)
            if f.shape[0] == 0:
                logging.warning(f"PrepareSTLMesh: no geometry left after cleaning/remeshing at batch index {i}; skipping")
                continue
            if target_faces > 0 and f.shape[0] > target_faces:
                v, f, _, _, _ = qem_decimate_simplify(v, f, target_faces)
            # Close the DC cracks, guarantee a 2-manifold shell, and reach the same weld
            # fixed point SaveOBJ uses on export so the mesh is watertight all the way out.
            v, f = _resolve_topology(v, f)
            for _ in range(8):
                welded_v, welded_f = _weld(v, f)
                if welded_v.shape[0] == v.shape[0]:
                    break
                v, f = _resolve_topology(welded_v, welded_f)
            v, f = _compact(v, f)
            if _has_nonmanifold_edges(f) or _has_boundary_edges(f):
                logging.warning(
                    f"PrepareSTLMesh: could not fully resolve topology at batch index {i}; the "
                    "mesh may not be watertight. Try a higher remesh_resolution.")
            v = v.to(device)
            f = f.to(device)
            out_v.append(v)
            out_f.append(f)
            v_counts.append(v.shape[0])
            f_counts.append(f.shape[0])

        if not out_v:
            raise ValueError("PrepareSTLMesh: no non-empty meshes to output")

        max_v = max(v_counts)
        max_f = max(f_counts)
        if len(out_v) == 1 and v_counts[0] == max_v and f_counts[0] == max_f:
            vertices = out_v[0].unsqueeze(0)
            faces = out_f[0].unsqueeze(0)
            vertex_counts = face_counts = None
        else:
            pad_v = max_v - min(v_counts)
            pad_f = max_f - min(f_counts)
            if pad_v == 0 and pad_f == 0:
                vertices = torch.stack(out_v, dim=0)
                faces = torch.stack(out_f, dim=0)
                vertex_counts = face_counts = None
            else:
                vertices = torch.zeros(len(out_v), max_v, 3, dtype=out_v[0].dtype, device=out_v[0].device)
                faces = torch.zeros(len(out_f), max_f, 3, dtype=out_f[0].dtype, device=out_f[0].device)
                for i, (v, f) in enumerate(zip(out_v, out_f)):
                    vertices[i, :v.shape[0]] = v
                    faces[i, :f.shape[0]] = f
                vertex_counts = torch.tensor(v_counts, dtype=torch.int64, device=out_v[0].device)
                face_counts = torch.tensor(f_counts, dtype=torch.int64, device=out_f[0].device)

        return IO.NodeOutput(Types.MESH(vertices, faces, vertex_counts=vertex_counts, face_counts=face_counts))
