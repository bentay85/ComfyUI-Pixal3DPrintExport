import { app } from "../../scripts/app.js";

const MODEL_CONFIG_DEFAULTS = {
  upDirection: "original",
  materialMode: "original",
  showSkeleton: false,
  gizmo: {
    enabled: false,
    mode: "translate",
    position: { x: 0, y: 0, z: 0 },
    rotation: { x: 0, y: 0, z: 0 },
    scale: { x: 1, y: 1, z: 1 }
  }
};

// SaveOBJ exports Z-up meshes for slicers while the Load3D viewer assumes the
// model declares its own up axis, so tell the viewer which way is up.
function syncPreviewUpDirection(node, standUp) {
  const current = node.properties["Model Config"] ?? {};
  node.properties["Model Config"] = {
    ...MODEL_CONFIG_DEFAULTS,
    ...current,
    upDirection: standUp ? "z" : "original",
    gizmo: { ...MODEL_CONFIG_DEFAULTS.gizmo, ...(current.gizmo ?? {}) }
  };
}

app.registerExtension({
  name: "Pixal3DPrint.SaveOBJ",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SaveOBJ") return;

    // Same preview widget SaveGLB gets; core's PREVIEW_3D factory builds it.
    nodeData.input.required.image = ["PREVIEW_3D"];

    // The core Comfy.SaveGLB extension owns the preview, the export menu and the
    // viewport state, and routes on the *static* node.constructor.comfyClass.
    // class_type is read from the prototype's comfyClass instead, which stays
    // "SaveOBJ", so the prompt still executes SaveOBJ on the backend.
    nodeType.comfyClass = "SaveGLB";
  },

  nodeCreated(node) {
    if (node.constructor.nodeData?.name !== "SaveOBJ") return;

    const standUp = node.widgets?.find((w) => w.name === "stand_up");
    if (!standUp) return;

    syncPreviewUpDirection(node, !!standUp.value);

    const originalCallback = standUp.callback;
    standUp.callback = function (...args) {
      syncPreviewUpDirection(node, !!standUp.value);
      return originalCallback?.apply(this, args);
    };
  }
});
