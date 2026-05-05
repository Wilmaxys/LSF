/**
 * Cœur du viewer 3D : scène Three.js + chargement VRM + lecture d'animations.
 *
 * Le viewer est paramétrable : on peut changer l'avatar VRM ou l'animation
 * .vrma à tout moment via loadAvatar() / loadAnimation(). Les méthodes prennent
 * une URL **ou** un objet { name, data } (utilisé pour le drag & drop).
 *
 * Si une animation a été produite pour un autre avatar, elle reste compatible :
 * createVRMAnimationClip() résout les noms de bones humanoïdes VRM standards
 * vers les nœuds three.js de l'avatar courant. Les bones cibles absents sont
 * simplement ignorés (warning console).
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils, type VRM } from "@pixiv/three-vrm";
import {
  createVRMAnimationClip,
  VRMAnimationLoaderPlugin,
  VRMLookAtQuaternionProxy,
  type VRMAnimation,
} from "@pixiv/three-vrm-animation";

export type LoadInput = string | { name: string; data: ArrayBuffer };

export class Viewer {
  readonly canvas: HTMLCanvasElement;
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;
  readonly controls: OrbitControls;
  readonly clock: THREE.Clock;
  readonly loader: GLTFLoader;

  private vrm: VRM | null = null;
  private mixer: THREE.AnimationMixer | null = null;
  private action: THREE.AnimationAction | null = null;
  private clip: THREE.AnimationClip | null = null;
  private speed = 1.0;
  private animationFrameId: number | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.clock = new THREE.Clock();

    // Scène
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x202028);

    // Lumières
    const hemi = new THREE.HemisphereLight(0xffffff, 0x444466, 1.2);
    this.scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 1.0);
    dir.position.set(1, 2, 1);
    this.scene.add(dir);

    // Sol (grille discrète)
    const grid = new THREE.GridHelper(10, 20, 0x444444, 0x2a2a2a);
    this.scene.add(grid);

    // Caméra
    this.camera = new THREE.PerspectiveCamera(
      40,
      canvas.clientWidth / canvas.clientHeight,
      0.1,
      100,
    );
    this.camera.position.set(0, 1.4, 2.5);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    // Controls
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 1.0, 0);
    this.controls.enableDamping = true;
    this.controls.update();

    // GLTFLoader avec plugins VRM
    this.loader = new GLTFLoader();
    this.loader.register((parser) => new VRMLoaderPlugin(parser));
    this.loader.register((parser) => new VRMAnimationLoaderPlugin(parser));

    // Resize handler
    window.addEventListener("resize", () => this.onResize());
  }

  /** Démarre la boucle de rendu. Idempotent. */
  start(): void {
    if (this.animationFrameId !== null) return;
    const tick = () => {
      this.animationFrameId = requestAnimationFrame(tick);
      this.render();
    };
    tick();
  }

  stop(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  private render(): void {
    const dt = this.clock.getDelta();
    if (this.mixer) this.mixer.update(dt);
    if (this.vrm) this.vrm.update(dt);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  private onResize(): void {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // ── Chargement avatar ─────────────────────────────────────────────────

  async loadAvatar(input: LoadInput): Promise<void> {
    const gltf = await this.parse(input);
    const vrm = (gltf.userData.vrm as VRM | undefined) ?? null;
    if (!vrm) {
      throw new Error(
        "Le fichier ne contient pas l'extension VRM. " +
          "Vérifiez qu'il s'agit bien d'un avatar .vrm.",
      );
    }

    // Disposer de l'ancien avatar
    if (this.vrm) {
      this.scene.remove(this.vrm.scene);
      VRMUtils.deepDispose(this.vrm.scene);
    }
    if (this.mixer) {
      this.mixer.stopAllAction();
      this.mixer = null;
      this.action = null;
    }

    VRMUtils.removeUnnecessaryVertices(vrm.scene);
    VRMUtils.combineSkeletons(vrm.scene);

    // Proxy quaternion pour que les pistes de lookAt soient pilotables
    if (vrm.lookAt) {
      const proxy = new VRMLookAtQuaternionProxy(vrm.lookAt);
      proxy.name = "lookAtQuaternionProxy";
      vrm.scene.add(proxy);
    }

    // VRM 0.x est en miroir (bones inversés) — three-vrm gère ça mais on cale
    // tout de même la rotation pour éviter les surprises.
    vrm.scene.rotation.y = Math.PI;

    this.scene.add(vrm.scene);
    this.vrm = vrm;

    console.info("[Viewer] Avatar chargé :", typeof input === "string" ? input : input.name);
    console.info(
      "[Viewer] VRM version :",
      vrm.meta?.metaVersion ?? "inconnue",
      "— bones :",
      vrm.humanoid ? Object.keys(vrm.humanoid.normalizedHumanBones).length : "n/a",
    );
  }

  // ── Chargement animation ─────────────────────────────────────────────

  async loadAnimation(input: LoadInput): Promise<void> {
    if (!this.vrm) {
      throw new Error("Charger un avatar avant de charger une animation.");
    }
    const gltf = await this.parse(input);

    const vrmAnimations = gltf.userData.vrmAnimations as
      | VRMAnimation[]
      | undefined;

    let clip: THREE.AnimationClip;
    if (vrmAnimations && vrmAnimations.length > 0) {
      // .vrma — retargeting auto sur l'avatar courant
      clip = createVRMAnimationClip(vrmAnimations[0], this.vrm);
      console.info("[Viewer] Animation .vrma chargée — retargeting auto");
    } else if (gltf.animations.length > 0) {
      // .glb avec animations standards — on ne fait pas de retargeting auto.
      // Pour qu'elles fonctionnent, les noms de bones du clip doivent
      // correspondre aux noms des nœuds dans la scène VRM.
      clip = gltf.animations[0];
      console.warn(
        "[Viewer] Animation .glb (non-VRM) chargée — pas de retargeting auto. " +
          "Les pistes seront appliquées telles quelles.",
      );
    } else {
      throw new Error(
        "Le fichier ne contient aucune animation (ni VRMC_vrm_animation, ni glTF.animations).",
      );
    }

    if (this.mixer) this.mixer.stopAllAction();
    this.mixer = new THREE.AnimationMixer(this.vrm.scene);
    this.action = this.mixer.clipAction(clip);
    this.action.setLoop(THREE.LoopRepeat, Infinity);
    this.action.play();
    this.action.timeScale = this.speed;
    this.clip = clip;

    console.info(
      "[Viewer] Animation prête :",
      typeof input === "string" ? input : input.name,
      "— durée :", clip.duration.toFixed(2), "s",
      "— pistes :", clip.tracks.length,
    );
  }

  // ── Contrôles de lecture ─────────────────────────────────────────────

  play(): void {
    this.action?.play();
    this.action && (this.action.paused = false);
  }
  pause(): void {
    if (this.action) this.action.paused = true;
  }
  togglePlayPause(): void {
    if (!this.action) return;
    this.action.paused = !this.action.paused;
    if (!this.action.paused) this.action.play();
  }
  isPlaying(): boolean {
    return this.action !== null && !this.action.paused;
  }
  setTime(seconds: number): void {
    if (this.action) this.action.time = seconds;
  }
  getTime(): number {
    return this.action?.time ?? 0;
  }
  getDuration(): number {
    return this.clip?.duration ?? 0;
  }

  setSpeed(speed: number): void {
    this.speed = speed;
    if (this.action) this.action.timeScale = speed;
  }
  getSpeed(): number {
    return this.speed;
  }

  hasAnimation(): boolean {
    return this.action !== null;
  }

  // ── Helpers ───────────────────────────────────────────────────────────

  private async parse(input: LoadInput) {
    if (typeof input === "string") {
      return this.loader.loadAsync(input);
    }
    return new Promise<Awaited<ReturnType<typeof this.loader.loadAsync>>>(
      (resolve, reject) => {
        this.loader.parse(
          input.data,
          "",
          (gltf) =>
            resolve(gltf as Awaited<ReturnType<typeof this.loader.loadAsync>>),
          reject,
        );
      },
    );
  }
}
