import logging
import os

import folder_paths
import torch

from comfy_api.latest import IO, Types

from .mesh_file import file3d_to_tensors, mesh_to_obj_bytes


def fit_for_print(v: torch.Tensor, stand_up: bool, model_height: float) -> torch.Tensor:
    """Orient and scale a mesh so it sits upright on the build plate.

    Slicers are Z-up while ComfyUI 3D files are Y-up, so `stand_up` rotates the
    model a quarter turn about X (+Y becomes +Z). `model_height` then rescales the
    model along the up axis, and the model is always dropped onto z=0.
    """
    if stand_up:
        v = torch.stack((v[:, 0], -v[:, 2], v[:, 1]), dim=-1)
    if model_height > 0:
        extent = float(v[:, 2].max() - v[:, 2].min())
        if extent > 0.0:
            v = v * (model_height / extent)
    return v - v[:, 2].min()


class SaveOBJ(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveOBJ",
            display_name="Save OBJ",
            search_aliases=["export obj", "save obj", "wavefront", "obj file"],
            category="Pixal3DPrintExport",
            is_output_node=True,
            description="Save a mesh (with optional vertex colors) to a Wavefront .obj file, ready for slicing.",
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
                    tooltip="Mesh or 3D file to save as OBJ.",
                ),
                IO.String.Input("filename_prefix", default="3d/ComfyUI"),
                IO.Boolean.Input(
                    "stand_up",
                    default=True,
                    tooltip="Rotate the model so it stands upright on the build plate instead of lying down.",
                ),
                IO.Float.Input(
                    "model_height",
                    default=0.0,
                    min=0.0,
                    max=10000.0,
                    tooltip="Scale the model to this height (in file units, mm for slicers). 0 keeps the original size.",
                ),
                IO.Int.Input(
                    "color_count",
                    default=0,
                    min=0,
                    max=256,
                    tooltip="Reduce the vertex colors to at most this many colors. 0 keeps every color.",
                ),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(
        cls,
        mesh: Types.MESH | Types.File3D,
        filename_prefix: str,
        stand_up: bool = True,
        model_height: float = 0.0,
        color_count: int = 0,
    ) -> IO.NodeOutput:
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        results = []

        if isinstance(mesh, Types.File3D):
            v, f = file3d_to_tensors(mesh)
            items = [(v, f, None)]
        else:
            items = []
            for i in range(mesh.vertices.shape[0]):
                vv = mesh.vertices[i]
                ff = mesh.faces[i]
                cc = mesh.vertex_colors[i] if mesh.vertex_colors is not None else None
                if mesh.vertex_counts is not None:
                    vc = int(mesh.vertex_counts[i].item())
                    fc = int(mesh.face_counts[i].item())
                    vv = vv[:vc]
                    ff = ff[:fc]
                    if cc is not None:
                        cc = cc[:vc]
                items.append((vv, ff, cc))

        for i, (vv, ff, cc) in enumerate(items):
            if vv.shape[0] == 0 or ff.shape[0] == 0:
                logging.warning(f"SaveOBJ: skipping empty mesh at batch index {i}")
                continue
            obj = mesh_to_obj_bytes(fit_for_print(vv, stand_up, model_height), ff, cc, color_count=color_count)
            fname = f"{filename}_{counter:05}_.obj"
            with open(os.path.join(full_output_folder, fname), "wb") as fh:
                fh.write(obj)
            results.append({"filename": fname, "subfolder": subfolder, "type": "output"})
            counter += 1

        return IO.NodeOutput(ui={"3d": results})
