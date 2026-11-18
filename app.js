// X-ray Mirror: webcam -> MediaPipe face + hand tracking -> BodyParts3D anatomy overlay.
// Everything is loaded from this folder; no frame or landmark ever leaves the page.
import * as THREE from "three";
import { GLTFLoader } from "./vendor/three/loaders/GLTFLoader.js";
import {
  FilesetResolver,
  FaceLandmarker,
  HandLandmarker,
  PoseLandmarker,
} from "./vendor/mediapipe/vision_bundle.mjs";
import { createBody, activeMaterial, stripMaterial } from "./body.js";
import { createEstimator, GROUP_LABEL } from "./activity.js";

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
const statusEl = $("#status"),
  frame = $("#frame"),
  canvas = $("#gl");
const setStatus = (t) => {
  statusEl.textContent = t;
};
const state = {
  style: "xray",
  layers: {
    skull: true,
    brain: true,
    muscles: true,
    eyes: true,
    hands: true,
    body: true,
    bodyMuscles: true,
  },
  fit: { scale: 1, y: 0, z: 0, tilt: 0 },
};
const debug = (window.__xray = {
  face: false,
  hands: 0,
  jawOpen: 0,
  reprojPx: null,
  frames: 0,
  error: null,
});

// MediaPipe's face-geometry camera: 63 degree vertical FOV at the origin, looking down -z, units cm.
const FACE_FOV = 63;
// A few canonical-face landmarks (cm) used only to self-check the camera model.
const CANON = {
  1: [0, -1.126865, 7.475604],
  10: [0, 8.261778, 4.481535],
  152: [0, -9.403378, 4.264492],
  33: [-4.445859, 2.663991, 3.173422],
  263: [4.445859, 2.663991, 3.173422],
};
const JAW_MAX = 0.42; // radians of mandible rotation at jawOpen = 1

// ---------- materials ----------
const XRAY = {
  skull: [0.62, 0.82, 1.0, 0.8],
  brain: [1.0, 0.42, 0.55, 0.3],
  muscles: [1.0, 0.3, 0.22, 0.6],
  eyes: [0.5, 1.0, 0.85, 0.45],
  hands: [0.7, 0.88, 1.0, 1.3],
  body: [0.62, 0.82, 1.0, 0.9],
};
const SOLID = {
  skull: 0xe9e0c9,
  brain: 0xe2a3a3,
  muscles: 0xb23a32,
  eyes: 0xf4f4f0,
  hands: 0xe9e0c9,
  body: 0xe9e0c9,
};
function xrayMaterial([r, g, b, gain]) {
  return new THREE.ShaderMaterial({
    uniforms: {
      color: { value: new THREE.Color(r, g, b) },
      gain: { value: gain },
    },
    vertexShader: `varying vec3 vN; varying vec3 vV;
      void main(){ vec4 mv = modelViewMatrix * vec4(position,1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `uniform vec3 color; uniform float gain; varying vec3 vN; varying vec3 vV;
      void main(){ float rim = 1.0 - abs(dot(normalize(vN), normalize(vV))); float a = (0.015 + pow(rim, 3.0) * 0.55) * gain; gl_FragColor = vec4(color * a, a); }`,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    side: THREE.DoubleSide,
    blending: THREE.CustomBlending,
    blendEquation: THREE.AddEquation,
    blendSrc: THREE.OneFactor,
    blendDst: THREE.OneFactor,
  });
}
const materials = {};
for (const k of Object.keys(XRAY))
  materials[k] = {
    xray: xrayMaterial(XRAY[k]),
    solid: new THREE.MeshStandardMaterial({
      color: SOLID[k],
      roughness: 0.75,
      metalness: 0,
      side: THREE.DoubleSide,
    }),
  };

// ---------- scene ----------
const renderer = new THREE.WebGLRenderer({
  canvas,
  alpha: true,
  antialias: true,
  premultipliedAlpha: true,
});
renderer.setClearColor(0x000000, 0);
renderer.autoClear = false;
const faceCam = new THREE.PerspectiveCamera(FACE_FOV, 1, 1, 10000);
const faceScene = new THREE.Scene();
const handCam = new THREE.OrthographicCamera(-1, 1, 1, -1, -4000, 4000);
const handScene = new THREE.Scene();
for (const sc of [faceScene, handScene]) {
  sc.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.6));
  const d = new THREE.DirectionalLight(0xffffff, 1.4);
  d.position.set(0.3, 0.8, 1);
  sc.add(d);
}
const headRoot = new THREE.Group();
headRoot.visible = false;
faceScene.add(headRoot); // tracked pose
const headFit = new THREE.Group();
headRoot.add(headFit); // user fine-tune
const jawPivot = new THREE.Group();
headFit.add(jawPivot);
const layerMeshes = {
  skull: [],
  brain: [],
  muscles: [],
  eyes: [],
  hands: [],
  body: [],
  bodyMuscles: [],
};
// brain regions that light up: own material pair each, shared with the brain inset
const BRAIN_COLD = [1.0, 0.42, 0.55];
const REGION_HOT = {
  m1_L: [1.0, 0.9, 0.35],
  m1_R: [1.0, 0.9, 0.35],
  s1_L: [0.4, 0.95, 1.0],
  s1_R: [0.4, 0.95, 1.0],
  visual: [0.75, 0.6, 1.0],
  broca: [0.5, 1.0, 0.6],
  cerebellum: [1.0, 0.65, 0.3],
};
const regionMats = {};
const LAYER_OF = {
  cranium: "skull",
  neck: "skull",
  jaw: "skull",
  brain: "brain",
  muscles: "muscles",
  eyes: "eyes",
};
let meta,
  handRigs = [];

async function loadModels() {
  const loader = new GLTFLoader();
  const [m, headGltf, handGltf] = await Promise.all([
    fetch("./models/meta.json").then((r) => r.json()),
    loader.loadAsync("./models/head.glb"),
    loader.loadAsync("./models/hand.glb"),
  ]);
  meta = m;
  const take = (gltf) => {
    const out = {};
    gltf.scene.traverse((o) => {
      if (o.isMesh) {
        o.geometry.computeVertexNormals();
        out[o.name] = o;
      }
    });
    return out;
  };
  const head = take(headGltf);
  jawPivot.position.fromArray(meta.head.jawHinge);
  for (const [name, mesh] of Object.entries(head)) {
    const layer = name.startsWith("brain") ? "brain" : LAYER_OF[name];
    mesh.frustumCulled = false;
    const region = name.startsWith("brain_") && name.slice(6);
    if (region) {
      const hot = REGION_HOT[region];
      mesh.userData.mats = regionMats[region] = {
        xray: meta.head.strips[region]
          ? stripMaterial(BRAIN_COLD, hot, 0.3, meta.head.strips[region])
          : activeMaterial(BRAIN_COLD, hot, 0.3),
        solid: new THREE.MeshStandardMaterial({
          color: SOLID.brain,
          emissive: new THREE.Color(...hot),
          emissiveIntensity: 0,
          roughness: 0.75,
          side: THREE.DoubleSide,
        }),
      };
    }
    layerMeshes[layer].push(mesh);
    if (name === "jaw") {
      mesh.position.fromArray(meta.head.jawHinge).negate();
      jawPivot.add(mesh);
    } else headFit.add(mesh);
  }
  // brain first, bone last, so the solid style still reads sensibly through toggles
  const hand = take(handGltf);
  for (let i = 0; i < 2; i++) {
    const rig = { group: new THREE.Group(), bones: [] };
    rig.group.visible = false;
    for (const b of meta.hand.bones) {
      const mesh = new THREE.Mesh(hand[b.name].geometry, materials.hands.xray);
      mesh.matrixAutoUpdate = false;
      mesh.frustumCulled = false;
      rig.group.add(mesh);
      rig.bones.push({ ...b, mesh });
      layerMeshes.hands.push(mesh);
    }
    handScene.add(rig.group);
    handRigs.push(rig);
  }
  applyStyle();
}
function applyStyle() {
  for (const [layer, meshes] of Object.entries(layerMeshes))
    for (const mesh of meshes) {
      mesh.material = (mesh.userData.mats ?? materials[layer])[state.style];
      mesh.userData.layerOn = state.layers[layer];
      mesh.visible = state.layers[layer];
    }
}

// ---------- face ----------
const tgt = {
    p: new THREE.Vector3(),
    q: new THREE.Quaternion(),
    s: new THREE.Vector3(),
  },
  tmpM = new THREE.Matrix4();
let faceSeen = false,
  jaw = 0;
// Far from the camera the face tracker loses the face; place the head from the pose's eyes and mouth instead.
const CANON_EYES = new THREE.Vector3(0, 2.66, 3.17); // midpoint of the outer eye corners, canonical cm
function headFromPose(lm, w, h) {
  if (!lm) return false;
  if (Math.min(...[3, 6, 9, 10].map((i) => lm[i].visibility ?? 1)) < 0.6)
    return false;
  const p = (i) =>
    new THREE.Vector3(
      (lm[i].x - 0.5) * w,
      (0.5 - lm[i].y) * h,
      -lm[i].z * w * 0.5, // pose depth on the face is noisy: trust it half
    );
  const eL = p(3),
    eR = p(6),
    eye = eL.clone().add(eR).multiplyScalar(0.5),
    mouth = p(9).add(p(10)).multiplyScalar(0.5);
  const x = eL.clone().sub(eR),
    d = x.length();
  if (d < 4) return false;
  x.normalize();
  const y = eye.clone().sub(mouth);
  y.addScaledVector(x, -y.dot(x)).normalize();
  const f = h / 2 / Math.tan(THREE.MathUtils.degToRad(FACE_FOV / 2));
  const Z = (f * 8.9) / d; // outer eye corners are about 8.9 cm apart
  tgt.q.setFromRotationMatrix(
    tmpM.makeBasis(x, y, new THREE.Vector3().crossVectors(x, y)),
  );
  tgt.p
    .set((eye.x / f) * Z, (eye.y / f) * Z, -Z)
    .sub(CANON_EYES.clone().applyQuaternion(tgt.q));
  tgt.s.set(1, 1, 1);
  return true;
}
function updateFace(res, w, h, poseLm) {
  const mat = res.facialTransformationMatrixes?.[0];
  let ok = !!mat;
  if (mat) tmpM.fromArray(mat.data).decompose(tgt.p, tgt.q, tgt.s);
  else ok = headFromPose(poseLm, w, h);
  debug.face = mat ? true : ok ? "pose" : false;
  headRoot.visible = ok;
  if (!ok) {
    faceSeen = false;
    return;
  }
  const k = faceSeen ? (mat ? 0.55 : 0.3) : 1; // light smoothing; snap on first sight
  headRoot.position.lerp(tgt.p, k);
  headRoot.quaternion.slerp(tgt.q, k);
  headRoot.scale.lerp(tgt.s, k);
  faceSeen = true;
  const open =
    res.faceBlendshapes?.[0]?.categories.find(
      (c) => c.categoryName === "jawOpen",
    )?.score ?? 0;
  jaw += ((debug.forceJaw ?? open) - jaw) * 0.6;
  debug.jawOpen = jaw;
  jawPivot.rotation.x = jaw * JAW_MAX;
  headFit.position.set(0, state.fit.y, state.fit.z);
  headFit.rotation.x = THREE.MathUtils.degToRad(state.fit.tilt);
  headFit.scale.setScalar(state.fit.scale);
  if (!mat) {
    debug.reprojPx = null;
    return;
  }
  // self-check: canonical points pushed through the raw matrix + camera should land on the 2D landmarks
  const lm = res.faceLandmarks[0],
    v = new THREE.Vector3();
  let err = 0,
    n = 0;
  for (const [i, c] of Object.entries(CANON)) {
    v.fromArray(c).applyMatrix4(tmpM).project(faceCam);
    err += Math.hypot(
      ((v.x + 1) / 2 - lm[i].x) * w,
      ((1 - v.y) / 2 - lm[i].y) * h,
    );
    n++;
  }
  debug.reprojPx = err / n;
}

// ---------- hands ----------
const V = () => new THREE.Vector3();
const basis = new THREE.Matrix4(),
  restB = new THREE.Matrix4(),
  T1 = new THREE.Matrix4(),
  T2 = new THREE.Matrix4();
function frameOf(a, lateral, sign, out) {
  // orthonormal (axis, side, dorsal); sign = -1 mirrors for a left hand
  const side = lateral.clone().addScaledVector(a, -a.dot(lateral)).normalize();
  const d = V().crossVectors(a, side).multiplyScalar(sign);
  return out.makeBasis(a, side, d);
}
function updateHand(rig, lm, isRight, w, h) {
  const P = lm.map(
    (l) => new THREE.Vector3((l.x - 0.5) * w, (0.5 - l.y) * h, -l.z * w),
  );
  const R = meta.hand.rest,
    r = (a) => new THREE.Vector3().fromArray(a);
  const sign = isRight ? 1 : -1;
  const W0 = r(R.wrist),
    R5 = r(R.mcp[1]),
    R17 = r(R.mcp[4]);
  // palm block: affine taking rest (wrist, index knuckle, little knuckle) onto the live landmarks
  const ur = R5.clone().sub(W0),
    vr = R17.clone().sub(W0),
    nr = V()
      .crossVectors(ur, vr)
      .normalize()
      .multiplyScalar(Math.sqrt(ur.length() * vr.length()));
  const u = P[5].clone().sub(P[0]),
    v = P[17].clone().sub(P[0]),
    n = V()
      .crossVectors(u, v)
      .normalize()
      .multiplyScalar(sign * Math.sqrt(u.length() * v.length()));
  const A = new THREE.Matrix4()
    .makeBasis(u, v, n)
    .multiply(new THREE.Matrix4().makeBasis(ur, vr, nr).invert());
  const palm = new THREE.Matrix4()
    .makeTranslation(P[0])
    .multiply(A)
    .multiply(new THREE.Matrix4().makeTranslation(W0.clone().negate()));
  const thick =
    Math.sqrt(u.length() * v.length()) / Math.sqrt(ur.length() * vr.length());
  const latR = R17.clone().sub(R5),
    lat = P[17].clone().sub(P[5]);
  for (const b of rig.bones) {
    if (b.kind === "carpal") {
      b.mesh.matrix.copy(palm);
      continue;
    }
    const f = b.finger,
      base = f === 0 ? 1 : 5 + 4 * (f - 1);
    const pr = r(b.prox),
      dr = r(b.dist);
    let p, d;
    if (b.kind === "mc") {
      p = f === 0 ? P[1] : pr.clone().applyMatrix4(palm);
      d = f === 0 ? P[2] : P[base];
    } else {
      const step =
        f === 0
          ? { prox: 1, dist: 2 }[b.kind]
          : { prox: 0, mid: 1, dist: 2 }[b.kind];
      p = P[base + step];
      d = P[base + step + 1];
    }
    const ar = dr.clone().sub(pr),
      a = d.clone().sub(p);
    const stretch = (a.length() / ar.length()) * (b.kind === "dist" ? 0.9 : 1); // tip landmark sits past the bone
    frameOf(ar.clone().normalize(), latR, 1, restB);
    frameOf(a.clone().normalize(), lat, sign, basis);
    T1.makeTranslation(p);
    T2.makeTranslation(pr.negate());
    b.mesh.matrix
      .copy(T1)
      .multiply(basis)
      .multiply(new THREE.Matrix4().makeScale(stretch, thick, thick))
      .multiply(restB.transpose())
      .multiply(T2);
  }
}
// finger movement per hand (0..1), for the hand part of the motor strip
const handAct = { L: 0, R: 0 },
  prevFingers = {};
function fingerMotion(lm, side, dt) {
  const size = Math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y) || 1;
  const now = [4, 8, 12, 16, 20].map((i) => [
    (lm[i].x - lm[0].x) / size,
    (lm[i].y - lm[0].y) / size,
  ]);
  const prev = prevFingers[side];
  prevFingers[side] = now;
  if (!prev || dt <= 0) return 0;
  let v = 0;
  now.forEach((p, i) => {
    v += Math.hypot(p[0] - prev[i][0], p[1] - prev[i][1]);
  });
  return Math.min(1, v / 5 / dt / 2.5); // 2.5 hand-lengths per second = full
}
function updateHands(res, w, h, dt) {
  const all = res.landmarks ?? [];
  const seen = {};
  all.forEach((lm, i) => {
    // MediaPipe labels handedness as if the frame were a mirrored selfie; ours is raw, so swap.
    const side = res.handedness[i][0].categoryName === "Left" ? "R" : "L";
    seen[side] = true;
    handAct[side] += (fingerMotion(lm, side, dt) - handAct[side]) * 0.3;
  });
  for (const s of ["L", "R"])
    if (!seen[s]) {
      handAct[s] *= 0.8;
      prevFingers[s] = null;
    }
  const list = state.layers.hands ? all : [];
  debug.hands = list.length;
  handRigs.forEach((rig, i) => {
    rig.group.visible = i < list.length;
    if (rig.group.visible)
      updateHand(
        rig,
        list[i],
        res.handedness[i][0].categoryName === "Left",
        w,
        h,
      );
  });
}

// ---------- body, activity estimate, brain inset ----------
let body, poseLm, insetRender;
const estimator = createEstimator();
const NO_POSE = Array.from({ length: 33 }, () => ({
  x: 0,
  y: 0,
  z: 0,
  visibility: 0,
}));
const STRIPS = ["m1_L", "m1_R", "s1_L", "s1_R"],
  PARTS = ["leg", "trunk", "arm", "hand", "face"];
function setBrain(b) {
  for (const k of STRIPS) {
    const m = regionMats[k];
    if (!m) continue;
    const a = PARTS.map((p) => b?.[k]?.[p] ?? 0);
    m.xray.uniforms.acts.value = a;
    m.solid.emissiveIntensity = Math.max(...a);
  }
  for (const k of ["visual", "broca", "cerebellum"]) {
    const m = regionMats[k];
    if (!m) continue;
    m.xray.uniforms.act.value = b?.[k] ?? 0;
    m.solid.emissiveIntensity = b?.[k] ?? 0;
  }
}
async function loadBody() {
  const [bm, gltf, fileset] = await Promise.all([
    fetch("./models/body-meta.json").then((r) => r.json()),
    new GLTFLoader().loadAsync("./models/body.glb"),
    FilesetResolver.forVisionTasks("./vendor/mediapipe/wasm"),
  ]);
  const b = createBody({
    gltf,
    meta: bm,
    scene: handScene,
    boneMaterials: materials.body,
    stick: params.has("stick"),
  });
  layerMeshes.body.push(...b.bones);
  layerMeshes.bodyMuscles.push(...b.muscles);
  applyStyle();
  const make = (delegate) =>
    PoseLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: "./models/pose_landmarker.task",
        delegate,
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  try {
    poseLm = await make("GPU");
  } catch {
    poseLm = await make("CPU");
  }
  body = b;
  debug.bodyReady = true;
}
function initInset() {
  const c = $("#brainView");
  const r = new THREE.WebGLRenderer({ canvas: c, alpha: true, antialias: true });
  r.setPixelRatio(Math.min(devicePixelRatio, 2));
  r.setSize(c.clientWidth || 200, c.clientHeight || 170, false);
  r.setClearColor(0x000000, 0);
  const scene = new THREE.Scene(),
    centre = new THREE.Vector3().fromArray(meta.head.brainCentre);
  const dim = xrayMaterial([1.0, 0.42, 0.55, 0.12]); // quiet background so the lit regions stand out
  for (const mesh of layerMeshes.brain) {
    const m = new THREE.Mesh(mesh.geometry, mesh.userData.mats?.xray ?? dim);
    m.frustumCulled = false;
    scene.add(m);
  }
  // seen from above and a little behind, nose towards the top: the left hemisphere is on the left
  const cam = new THREE.PerspectiveCamera(32, c.width / c.height, 1, 200);
  cam.up.set(0, 0, 1);
  cam.position.copy(centre).add(new THREE.Vector3(0, 33, -11));
  cam.lookAt(centre);
  insetRender = () => r.render(scene, cam);
}
const barsEl = $("#bars"),
  brainListEl = $("#brainList");
const SIDE_NAME = { L: "left", R: "right" };
function updatePanel(est) {
  const top = Object.entries(est.muscles)
    .filter(([, v]) => v > 0.08)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 7);
  barsEl.innerHTML = top.length
    ? top
        .map(([k, v]) => {
          const [g, s] = [k.slice(0, -2), k.slice(-1)];
          return `<div class="bar"><span>${GROUP_LABEL[g] ?? g} <i>${s}</i></span><b><u style="width:${Math.round(v * 100)}%"></u></b></div>`;
        })
        .join("")
    : `<div class="none">${debug.body ? "Resting" : "Step back so your body is in view"}</div>`;
  const lines = [];
  for (const hemi of ["L", "R"]) {
    const a = est.brain[`m1_${hemi}`];
    const [part, v] = Object.entries(a).sort((x, y) => y[1] - x[1])[0];
    if (v > 0.3) {
      const other = SIDE_NAME[hemi === "L" ? "R" : "L"];
      lines.push(
        `Motor + sensory strip ${hemi} · ${part === "trunk" || part === "face" ? part : other + " " + part}`,
      );
    }
  }
  if (est.brain.cerebellum > 0.3) lines.push("Cerebellum · coordination");
  if (est.brain.broca > 0.15) lines.push("Speech area (Broca)");
  if (est.brain.visual > 0.15) lines.push("Visual cortex");
  brainListEl.innerHTML = lines.map((l) => `<div>${l}</div>`).join("");
}

// ---------- source + loop ----------
let source,
  srcW = 0,
  srcH = 0,
  faceLm,
  handLm;
function layout() {
  const s = Math.min(innerWidth / srcW, innerHeight / srcH);
  frame.style.width = srcW * s + "px";
  frame.style.height = srcH * s + "px";
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(srcW * s, srcH * s, false);
  faceCam.aspect = srcW / srcH;
  faceCam.updateProjectionMatrix();
  handCam.left = -srcW / 2;
  handCam.right = srcW / 2;
  handCam.top = srcH / 2;
  handCam.bottom = -srcH / 2;
  handCam.updateProjectionMatrix();
}
async function openSource() {
  const src = params.get("src");
  if (src && /\.(jpe?g|png|webp)$/i.test(src)) {
    source = new Image();
    await new Promise((ok, fail) => {
      source.onload = ok;
      source.onerror = () => fail(new Error("could not load " + src));
      source.src = src;
    });
    srcW = source.naturalWidth;
    srcH = source.naturalHeight;
  } else {
    source = document.createElement("video");
    source.playsInline = true;
    source.muted = true;
    if (src) {
      source.src = src;
      source.loop = true;
    } else
      source.srcObject = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });
    await source.play();
    srcW = source.videoWidth;
    srcH = source.videoHeight;
  }
  if (params.has("nomirror")) frame.classList.add("unmirrored");
  frame.prepend(source);
  layout();
  addEventListener("resize", layout);
}
async function createTrackers() {
  const fileset = await FilesetResolver.forVisionTasks(
    "./vendor/mediapipe/wasm",
  );
  const make = (delegate) =>
    Promise.all([
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: "./models/face_landmarker.task",
          delegate,
        },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      }),
      HandLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: "./models/hand_landmarker.task",
          delegate,
        },
        runningMode: "VIDEO",
        numHands: 2,
      }),
    ]);
  try {
    [faceLm, handLm] = await make("GPU");
  } catch {
    [faceLm, handLm] = await make("CPU");
  }
}
let lastTs = 0;
function tick() {
  requestAnimationFrame(tick);
  renderFrame();
}
// Test hook (?test): render frames on demand and save the composite through the dev server.
if (params.has("test"))
  debug.shot = async (name, frames = 6) => {
    for (let i = 0; i < frames; i++) renderFrame();
    const c = document.createElement("canvas");
    c.width = srcW;
    c.height = srcH;
    const g = c.getContext("2d");
    g.drawImage(source, 0, 0, srcW, srcH);
    g.fillStyle = `rgba(3,9,20,${$("#tint").style.opacity || 0.6})`;
    g.fillRect(0, 0, srcW, srcH);
    g.drawImage(canvas, 0, 0, srcW, srcH);
    const blob = await new Promise((r) => c.toBlob(r, "image/png"));
    await fetch(`/__shot/${name}.png`, { method: "POST", body: blob });
    return JSON.stringify({ ...debug, shot: undefined });
  };
let estT = 0;
function renderFrame() {
  if (source.readyState !== undefined && source.readyState < 2) return;
  const t0 = performance.now();
  const ts = Math.max(t0, lastTs + 1);
  // ?test can pin the time step so a few frames are enough for the estimate to settle
  const dt = debug.fakeDt ?? (lastTs ? (ts - lastTs) / 1000 : 0);
  lastTs = ts;
  estT += dt;
  let est;
  try {
    const bodyOn = state.layers.body || state.layers.bodyMuscles;
    let pose = null;
    if (bodyOn && (debug.forcePose || poseLm)) {
      const r = debug.forcePose ?? poseLm.detectForVideo(source, ts);
      if (r.landmarks?.[0]) {
        pose = r;
        // the model guesses joints that are outside the picture: treat those as not seen
        r.landmarks[0].forEach((l, i) => {
          if (l.x < -0.02 || l.x > 1.02 || l.y < -0.02 || l.y > 1.02) {
            l.visibility = 0;
            if (r.worldLandmarks?.[0]) r.worldLandmarks[0][i].visibility = 0;
          }
        });
      }
    }
    const faceRes = faceLm.detectForVideo(source, ts);
    // The face tracker is the more reliable of the two. A pose whose eyes are not where the
    // tracked face has them is the pose model guessing at a body it cannot see: drop it.
    const fl = faceRes.faceLandmarks?.[0];
    debug.poseFit = null;
    debug.lastPose = pose?.landmarks[0];
    if (pose && fl && !debug.forcePose) {
      const l = pose.landmarks[0];
      const d = (a, b) => Math.hypot((a.x - b.x) * srcW, (a.y - b.y) * srcH);
      // 263 / 33 = outer corner of the subject's left / right eye, as pose 3 / 6
      debug.poseFit = (d(l[3], fl[263]) + d(l[6], fl[33])) / 2 / d(fl[263], fl[33]);
      if (debug.poseFit > 0.5) pose = null;
    }
    updateFace(faceRes, srcW, srcH, pose?.landmarks[0]);
    updateHands(handLm.detectForVideo(source, ts), srcW, srcH, dt);
    debug.body = body ? body.update(pose?.landmarks[0], srcW, srcH) : false;
    est = estimator.update(pose?.worldLandmarks?.[0] ?? NO_POSE, estT, {
      jawOpen: jaw,
      hand_L: handAct.L,
      hand_R: handAct.R,
    });
    debug.est = est;
    body?.setActivation(est.muscles);
    setBrain(est.brain);
  } catch (e) {
    debug.error = String(e);
  }
  renderer.clear();
  renderer.render(faceScene, faceCam);
  renderer.clearDepth();
  renderer.render(handScene, handCam);
  insetRender?.();
  debug.frames++;
  debug.ms = debug.ms
    ? debug.ms + (performance.now() - t0 - debug.ms) * 0.1
    : performance.now() - t0;
  if (est && debug.frames % 6 === 0) updatePanel(est);
  if (debug.frames % 15 === 0) {
    const seen = [
      debug.face && "face",
      debug.body && "body",
      debug.hands && `${debug.hands} hand${debug.hands > 1 ? "s" : ""}`,
    ].filter(Boolean);
    setStatus(
      seen.length
        ? `Tracking ${seen.join(" + ")}`
        : "Show your face, a hand or your whole body",
    );
  }
}
async function start() {
  $("#start").hidden = true;
  try {
    setStatus("Opening camera…");
    await openSource();
    setStatus("Loading tracker…");
    tick();
  } catch (e) {
    debug.error = String(e);
    setStatus(
      e.name === "NotAllowedError"
        ? "Camera permission was denied"
        : "Could not start: " + e.message,
    );
    $("#start").hidden = false;
  }
}

// ---------- UI ----------
try {
  Object.assign(
    state.fit,
    JSON.parse(localStorage.getItem("xray-fit") || "{}"),
  );
} catch {}
const fitInputs = document.querySelectorAll("[data-fit]");
const syncFit = () =>
  fitInputs.forEach((i) => {
    i.value = state.fit[i.dataset.fit];
  });
syncFit();
fitInputs.forEach((i) =>
  i.addEventListener("input", () => {
    state.fit[i.dataset.fit] = +i.value;
    try {
      localStorage.setItem("xray-fit", JSON.stringify(state.fit));
    } catch {}
  }),
);
$("#resetFit").onclick = () => {
  state.fit = { scale: 1, y: 0, z: 0, tilt: 0 };
  syncFit();
  try {
    localStorage.removeItem("xray-fit");
  } catch {}
};
document.querySelectorAll("[data-layer]").forEach((i) =>
  i.addEventListener("change", () => {
    state.layers[i.dataset.layer] = i.checked;
    applyStyle();
  }),
);
document.querySelectorAll("[data-style]").forEach((b) =>
  b.addEventListener("click", () => {
    state.style = b.dataset.style;
    document
      .querySelectorAll("[data-style]")
      .forEach((x) => x.classList.toggle("on", x === b));
    applyStyle();
  }),
);
$("#dim").addEventListener("input", (e) => {
  $("#tint").style.opacity = e.target.value;
});
$("#startBtn").onclick = start;

(async () => {
  try {
    await Promise.all([loadModels(), createTrackers()]);
    initInset();
    // the body layer loads on its own: if it fails, the head and hands still work
    loadBody().catch((e) => {
      debug.bodyError = String(e);
    });
    if (params.get("src")) start();
    else {
      setStatus("Ready");
      $("#start").hidden = false;
    }
  } catch (e) {
    debug.error = String(e);
    setStatus("Failed to load: " + e.message);
  }
})();
