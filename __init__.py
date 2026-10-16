from comfy_api.latest import ComfyExtension, IO
from typing_extensions import override

from .nodes_prepare_mesh import PrepareSTLMesh
from .nodes_save_obj import SaveOBJ

WEB_DIRECTORY = "./web"


class Pixal3DPrintExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [PrepareSTLMesh, SaveOBJ]


async def comfy_entrypoint() -> Pixal3DPrintExtension:
    return Pixal3DPrintExtension()
