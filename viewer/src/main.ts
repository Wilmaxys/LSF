/**
 * Point d'entrée du viewer LSF.
 *
 * Workflow :
 *   1. Lire les paramètres URL (?avatar=… &animation=…)
 *   2. Initialiser la scène Three.js
 *   3. Charger l'avatar VRM par défaut
 *   4. Charger l'animation par défaut (si fournie)
 *   5. Brancher l'UI (play/pause, timeline, vitesse)
 *   6. Activer le drag & drop pour charger d'autres avatars/animations à la volée
 */
import { Viewer } from "./viewer";
import { setupUI } from "./ui";
import { setupDragAndDrop } from "./loader";

const DEFAULT_AVATAR = "avatars/default.vrm";
const DEFAULT_ANIMATION: string | null = null;

const params = new URLSearchParams(window.location.search);
const avatarUrl = params.get("avatar") ?? DEFAULT_AVATAR;
const animationUrl = params.get("animation") ?? DEFAULT_ANIMATION;

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const statusEl = document.getElementById("status") as HTMLDivElement;

const viewer = new Viewer(canvas);

setStatus("Chargement de l'avatar…");
viewer
  .loadAvatar(avatarUrl)
  .then(() => {
    setStatus(`Avatar chargé : ${avatarUrl}`);

    if (animationUrl) {
      setStatus("Chargement de l'animation…");
      return viewer
        .loadAnimation(animationUrl)
        .then(() => setStatus(`Prêt — ${avatarUrl} + ${animationUrl}`));
    }

    setStatus(
      `Avatar prêt : ${avatarUrl}. Pas d'animation chargée — déposez un .vrma pour en jouer une.`,
      "warning",
    );
    return undefined;
  })
  .catch((err: unknown) => {
    console.error(err);
    const msg =
      err instanceof Error ? err.message : "Erreur inconnue de chargement";
    setStatus(`Erreur : ${msg}`, "error");
  });

setupUI(viewer);
setupDragAndDrop(viewer, setStatus);

viewer.start();

// ─────────────────────────────────────────────────────────────────────────────

function setStatus(
  text: string,
  level: "info" | "warning" | "error" = "info",
): void {
  statusEl.textContent = text;
  statusEl.className = level === "info" ? "" : level;
}
