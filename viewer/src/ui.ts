/**
 * UI : play/pause, timeline, contrôle de vitesse.
 *
 * Mise à jour de la timeline et du timestamp à chaque frame d'animation
 * (callback installé sur requestAnimationFrame).
 */
import { Viewer } from "./viewer";

const SPEED_PRESETS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0];

export function setupUI(viewer: Viewer): void {
  const btnPlay = document.getElementById("btn-play") as HTMLButtonElement;
  const btnSpeedDown = document.getElementById(
    "btn-speed-down",
  ) as HTMLButtonElement;
  const btnSpeedUp = document.getElementById(
    "btn-speed-up",
  ) as HTMLButtonElement;
  const speedLabel = document.getElementById("speed-label") as HTMLSpanElement;
  const timeline = document.getElementById("timeline") as HTMLInputElement;
  const timestamp = document.getElementById("timestamp") as HTMLSpanElement;

  let speedIdx = SPEED_PRESETS.indexOf(1.0);
  let userScrubbing = false;

  // ─ Play / pause ─────────────────────────────────────────────────────
  btnPlay.addEventListener("click", () => {
    viewer.togglePlayPause();
    btnPlay.textContent = viewer.isPlaying() ? "❙❙" : "▶︎";
  });

  // Espace pour play/pause
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && e.target === document.body) {
      e.preventDefault();
      viewer.togglePlayPause();
      btnPlay.textContent = viewer.isPlaying() ? "❙❙" : "▶︎";
    }
  });

  // ─ Vitesse ───────────────────────────────────────────────────────────
  btnSpeedDown.addEventListener("click", () => {
    speedIdx = Math.max(0, speedIdx - 1);
    applySpeed();
  });
  btnSpeedUp.addEventListener("click", () => {
    speedIdx = Math.min(SPEED_PRESETS.length - 1, speedIdx + 1);
    applySpeed();
  });
  function applySpeed(): void {
    const s = SPEED_PRESETS[speedIdx];
    viewer.setSpeed(s);
    speedLabel.textContent = `${s.toFixed(2)}×`;
  }

  // ─ Timeline (scrubbing) ──────────────────────────────────────────────
  timeline.addEventListener("input", () => {
    userScrubbing = true;
    const ratio = parseFloat(timeline.value);
    const dur = viewer.getDuration();
    if (dur > 0) viewer.setTime(ratio * dur);
  });
  timeline.addEventListener("change", () => {
    userScrubbing = false;
  });

  // ─ Boucle de rafraîchissement de l'UI ────────────────────────────────
  function tick(): void {
    requestAnimationFrame(tick);

    const ready = viewer.hasAnimation();
    btnPlay.disabled = !ready;
    timeline.disabled = !ready;

    if (ready && !userScrubbing) {
      const dur = viewer.getDuration();
      const t = viewer.getTime();
      timeline.value = dur > 0 ? String(t / dur) : "0";
      timestamp.textContent = `${t.toFixed(1)} / ${dur.toFixed(1)} s`;
    } else if (!ready) {
      timestamp.textContent = "0.0 / 0.0 s";
    }
  }
  tick();
}
