/**
 * Drag & drop : déposer un fichier .vrm, .vrma ou .glb sur la page pour le
 * charger.
 *
 * Heuristique de classification :
 *   - .vrm                              → avatar (loadAvatar)
 *   - .vrma                             → animation (loadAnimation)
 *   - .glb / .gltf : on inspecte le binaire :
 *       - extension VRMC_vrm présente   → avatar
 *       - extension VRMC_vrm_animation  → animation
 *       - sinon                         → animation par défaut (on essaie loadAnimation,
 *                                         peut échouer si le rig est non-VRM)
 */
import { Viewer } from "./viewer";

export type StatusFn = (
  msg: string,
  level?: "info" | "warning" | "error",
) => void;

export function setupDragAndDrop(viewer: Viewer, setStatus: StatusFn): void {
  const overlay = document.getElementById("drop-overlay") as HTMLDivElement;
  const target = document.getElementById("app") as HTMLDivElement;

  let dragDepth = 0;

  target.addEventListener("dragenter", (e) => {
    e.preventDefault();
    dragDepth++;
    overlay.classList.add("visible");
  });

  target.addEventListener("dragover", (e) => {
    e.preventDefault();
  });

  target.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) overlay.classList.remove("visible");
  });

  target.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    overlay.classList.remove("visible");

    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    void handleFile(file, viewer, setStatus);
  });
}

async function handleFile(
  file: File,
  viewer: Viewer,
  setStatus: StatusFn,
): Promise<void> {
  const ext = file.name.toLowerCase().split(".").pop() ?? "";
  setStatus(`Lecture de ${file.name}…`);

  let buffer: ArrayBuffer;
  try {
    buffer = await file.arrayBuffer();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setStatus(`Erreur de lecture : ${msg}`, "error");
    return;
  }

  const kind = classify(ext, buffer);

  try {
    if (kind === "avatar") {
      await viewer.loadAvatar({ name: file.name, data: buffer });
      setStatus(`Avatar chargé : ${file.name}`);
    } else if (kind === "animation") {
      await viewer.loadAnimation({ name: file.name, data: buffer });
      setStatus(`Animation chargée : ${file.name}`);
    } else {
      setStatus(
        `Fichier ${file.name} non reconnu — déposez un .vrm, .vrma ou .glb`,
        "warning",
      );
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setStatus(`Erreur : ${msg}`, "error");
    console.error(err);
  }
}

function classify(
  ext: string,
  buffer: ArrayBuffer,
): "avatar" | "animation" | "unknown" {
  if (ext === "vrm") return "avatar";
  if (ext === "vrma") return "animation";
  if (ext === "glb" || ext === "gltf") {
    // Pour un .glb/.gltf, on inspecte le JSON pour décider.
    return classifyGltfBinary(buffer);
  }
  return "unknown";
}

/**
 * Inspecte les premiers Ko d'un .glb pour trouver l'extension VRM.
 *
 * Format glTF binaire (.glb) :
 *   12 bytes magic header
 *   4 bytes JSON chunk length, 4 bytes chunk type, JSON UTF-8
 *   ...
 * On lit le JSON header et on regarde extensions / extensionsUsed.
 */
function classifyGltfBinary(buffer: ArrayBuffer): "avatar" | "animation" | "unknown" {
  try {
    const view = new DataView(buffer);
    const magic = view.getUint32(0, true);
    if (magic !== 0x46546c67 /* "glTF" */) {
      return "unknown";
    }
    const jsonChunkLength = view.getUint32(12, true);
    const jsonChunkType = view.getUint32(16, true);
    if (jsonChunkType !== 0x4e4f534a /* "JSON" */) return "unknown";
    const jsonBytes = new Uint8Array(buffer, 20, jsonChunkLength);
    const jsonText = new TextDecoder().decode(jsonBytes);
    const json = JSON.parse(jsonText) as {
      extensions?: Record<string, unknown>;
      extensionsUsed?: string[];
    };
    const usedSet = new Set(json.extensionsUsed ?? []);
    const hasKey = (k: string) =>
      usedSet.has(k) || (json.extensions !== undefined && k in json.extensions);

    if (hasKey("VRMC_vrm_animation")) return "animation";
    if (hasKey("VRMC_vrm") || hasKey("VRM")) return "avatar";
  } catch (err) {
    console.warn("[loader] Échec d'inspection du glTF :", err);
  }
  // Par défaut, on tente animation pour les .glb sans extension VRM
  return "animation";
}
