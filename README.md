# ComfyUI-Pixal3DPrintExport

ComfyUI nodes that take a generated 3D mesh (Pixal3D, TRELLIS, or any mesh / 3D file) and turn it
into a **watertight, 2-manifold, print-ready OBJ** — optionally keeping the model's vertex colors.

Meshes coming out of generative 3D pipelines are usually non-manifold, full of cracks and
self-intersections, and far too dense for a slicer. These nodes clean the geometry, remesh it into
a closed 2-manifold shell, decimate it to a sane triangle budget, and export a Z-up OBJ with a
scaled height and a reduced color palette.

## Nodes

Both nodes appear under the **Pixal3DPrintExport** category
(right-click canvas → Add Node → Pixal3DPrintExport).

### Prepare Mesh (3D Print) — `PrepareSTLMesh`

Cleans, remeshes (watertight) and decimates a mesh. Outputs a `MESH` (geometry only; it drops any
incoming vertex colors).

| input | type | default | description |
| --- | --- | --- | --- |
| `mesh` | MESH / File3D (STL, GLB, GLTF, OBJ, Any) | — | Mesh or 3D file to clean up. |
| `remesh_resolution` | INT (0–1024) | `256` | Dual Contouring voxel resolution along the longest axis. `0` disables remeshing. |
| `target_faces` | INT (0–10,000,000) | `100000` | Decimate to this face count. `0` disables decimation. |

**Output:** `mesh` — cleaned, watertight mesh ready for texturing or printing.

### Save OBJ — `SaveOBJ`

Saves a mesh (with optional vertex colors) to a Wavefront `.obj` file, ready for slicing. This is
an output node and also shows a 3D preview in the node.

| input | type | default | description |
| --- | --- | --- | --- |
| `mesh` | MESH / File3D (STL, GLB, GLTF, OBJ, Any) | — | Mesh or 3D file to save as OBJ. |
| `filename_prefix` | STRING | `3d/ComfyUI` | Output path prefix under `ComfyUI/output/`. |
| `stand_up` | BOOLEAN | `true` | Rotate the model so it stands upright on the build plate. |
| `model_height` | FLOAT (0–10000) | `100.0` | Scale the model to this height (file units, mm for slicers). `0` keeps the original size. |
| `color_count` | INT (0–256) | `0` | Reduce vertex colors to at most this many colors. `0` keeps every color. |

**Output:** writes `output/<prefix>_<counter>_.obj` and shows it in the preview.

## Installation

1. Clone into your ComfyUI custom nodes folder:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/bentay85/ComfyUI-Pixal3DPrintExport.git
   ```

2. Restart ComfyUI. The nodes show up under **Pixal3DPrintExport**.

There is nothing else to install: the nodes only use ComfyUI's bundled `torch`, `numpy` and `scipy`.

## Usage

A typical graph:

```
image/video → 3D model (Pixal3D / TRELLIS) → Prepare Mesh (3D Print) → PaintMesh → Save OBJ
```

- Run the generated mesh through **Prepare Mesh (3D Print)** to get a watertight shell.
- Optionally texture it (e.g. `PaintMesh`) — that only adds vertex colors, it does not change the
  geometry.
- Feed the result into **Save OBJ** to orient, scale and export it.

`remesh_resolution` controls how much detail survives: higher values keep more shape detail at the
cost of more triangles and time. `target_faces` then caps the final triangle count so the file stays
manageable for a slicer. If your mesh is already clean and you only want to re-orient/rescale it, set
`remesh_resolution` and `target_faces` to `0`.

Example workflows are collected in the [`workflows/`](workflows) folder.

## How it works

**Prepare Mesh (3D Print)**

1. **Clean** the input with ComfyUI's mesh cleaner (welds near-coincident vertices and drops
   degenerate, duplicate and unused faces).
2. **Remesh** with Manifold Dual Contouring over a narrow-band unsigned distance field. This is the
   robust step: it ignores self-intersections and messy topology and rebuilds a closed 2-manifold
   shell. It can still leave sub-voxel *cracks* where the distance-field layers pinch.
3. **Decimate** to `target_faces` with quadric error metrics (QEM).
4. **Resolve topology:** cap every open boundary loop with a centroid fan, and repair non-manifold
   (4-faces-per-edge) seams by splitting the offending corners and nudging the duplicates apart, so
   the shell becomes watertight and 2-manifold.
5. **Weld to a fixed point:** the vertices are snapped to the exact grid `Save OBJ` uses on export,
   and steps 4–5 repeat until welding changes nothing. This guarantees the mesh cannot be broken
   again by the export-time weld.

**Save OBJ**

- **Orient / scale:** ComfyUI meshes are Y-up while slicers are Z-up, so `stand_up` rotates the model
  a quarter turn about X, `model_height` rescales the up axis, and the model is dropped onto `z = 0`.
- **Export:** triangle faces are written with 1-based indices; vertex colors are converted from linear
  to sRGB.
- **Color reduction:** `color_count` quantizes the palette in **CIELAB** (so it is perceptually even)
  with area-weighted farthest-point seeding plus Lloyd k-means iterations, so the dominant colors —
  not dense noisy patches — drive the palette.

Because generative meshes are not deterministic even for the same input, both nodes re-check the
result and iterate until the shell is clean rather than assuming the first pass succeeds.

## Notes and limitations

- **Prepare Mesh closes every boundary loop**, including openings that were intentional, because the
  goal is a printable watertight part. If you need to keep a model open, skip this node.
- If a shell still cannot be made watertight or 2-manifold, the node logs a warning naming the batch
  item instead of failing silently; raise `remesh_resolution` or clean the input and run it again.
- Very large inputs (millions of faces) can be slow and VRAM-heavy, mostly during remeshing and QEM
  decimation.
- `target_faces` may not be hit exactly; decimation stops when the error budget is exhausted.
- No extra dependencies: the nodes use only ComfyUI's bundled `torch`, `numpy` and `scipy`.

## Credits

These nodes were **vibe coded** with [opencode](https://opencode.ai) and **DeepSeek V4.1 Flash**.
