// Full-body layer: fits the BodyParts3D skeleton and muscle groups to MediaPipe pose landmarks.
// Works in the same pixel space as the hand bones (x right, y up, z towards the camera).
import * as THREE from "three";

const V = (a) => (a ? new THREE.Vector3().fromArray(a) : new THREE.Vector3());
const X = new THREE.Vector3(1, 0, 0);
const M = () => new THREE.Matrix4();

// rim-light X-ray shader whose colour and brightness follow an activation uniform
const VERT = `varying vec3 vN; varying vec3 vV; varying float vY;
  void main(){ vec4 mv = modelViewMatrix * vec4(position,1.0); vN = normalize(normalMatrix * normal);
    vV = normalize(-mv.xyz); vY = position.y; gl_Position = projectionMatrix * mv; }`;
const additive = {
  transparent: true,
  depthWrite: false,
  depthTest: false,
  side: THREE.DoubleSide,
  blending: THREE.CustomBlending,
  blendEquation: THREE.AddEquation,
  blendSrc: THREE.OneFactor,
  blendDst: THREE.OneFactor,
};
export function activeMaterial(cold, hot, gain) {
  return new THREE.ShaderMaterial({
    uniforms: {
      cold: { value: new THREE.Color(...cold) },
      hot: { value: new THREE.Color(...hot) },
      gain: { value: gain },
      act: { value: 0 },
    },
    vertexShader: VERT,
    fragmentShader: `uniform vec3 cold; uniform vec3 hot; uniform float gain; uniform float act;
      varying vec3 vN; varying vec3 vV;
      void main(){ float rim = 1.0 - abs(dot(normalize(vN), normalize(vV)));
        float a = (0.015 + pow(rim, 3.0) * 0.55) * gain * (1.0 + act * 4.0) + act * 0.12;
        gl_FragColor = vec4(mix(cold, hot, act) * a, a); }`,
    ...additive,
  });
}
// motor / sensory strip: five body parts laid out along its height (leg at the top, face at the bottom)
export function stripMaterial(cold, hot, gain, [y0, y1]) {
  return new THREE.ShaderMaterial({
    uniforms: {
      cold: { value: new THREE.Color(...cold) },
      hot: { value: new THREE.Color(...hot) },
      gain: { value: gain },
      y0: { value: y0 },
      y1: { value: y1 },
      acts: { value: [0, 0, 0, 0, 0] },
    },
    vertexShader: VERT,
    fragmentShader: `uniform vec3 cold; uniform vec3 hot; uniform float gain; uniform float y0; uniform float y1;
      uniform float acts[5]; varying vec3 vN; varying vec3 vV; varying float vY;
      float bump(float t, float c, float w){ float d = (t - c) / w; return exp(-d * d); }
      void main(){ float rim = 1.0 - abs(dot(normalize(vN), normalize(vV)));
        float t = clamp((y1 - vY) / (y1 - y0), 0.0, 1.0);
        float act = max(max(acts[0] * bump(t, 0.06, 0.08), acts[1] * bump(t, 0.2, 0.06)),
          max(max(acts[2] * bump(t, 0.35, 0.08), acts[3] * bump(t, 0.55, 0.1)), acts[4] * bump(t, 0.84, 0.13)));
        float a = (0.015 + pow(rim, 3.0) * 0.55) * gain * (1.0 + act * 4.0) + act * 0.12;
        gl_FragColor = vec4(mix(cold, hot, act) * a, a); }`,
    ...additive,
  });
}

const MUSCLE_COLD = [0.95, 0.22, 0.18],
  MUSCLE_HOT = [1.0, 0.9, 0.4];
const SEGMENT_ENDS = {
  upperarm: ["shoulder", "elbow"],
  forearm: ["elbow", "wrist"],
  thigh: ["hip", "knee"],
  shin: ["knee", "ankle"],
  foot: ["heel", "toe"],
};
const IDX = {
  shoulder: [11, 12],
  elbow: [13, 14],
  wrist: [15, 16],
  hip: [23, 24],
  knee: [25, 26],
  ankle: [27, 28],
  heel: [29, 30],
  toe: [31, 32],
};
const STICK = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [27, 29],
  [29, 31],
  [24, 26],
  [26, 28],
  [28, 30],
  [30, 32],
];

function frameOf(a, lateral, out) {
  const side = lateral.clone().addScaledVector(a, -a.dot(lateral));
  if (side.lengthSq() < 1e-8) side.set(a.y, -a.x, 0);
  side.normalize();
  return out.makeBasis(a, side, V().crossVectors(a, side));
}

export function createBody({ gltf, meta, scene, boneMaterials, stick }) {
  const group = new THREE.Group();
  group.visible = false;
  scene.add(group);
  const bones = [],
    muscles = [],
    parts = {}; // segment name -> { meshes, vis }
  const act = {}; // muscle key (deltoid_L) -> material pair
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    o.geometry.computeVertexNormals();
    o.matrixAutoUpdate = false;
    o.frustumCulled = false;
    let seg;
    if (o.name.startsWith("b_")) {
      seg = o.name.slice(2);
      o.userData.mats = boneMaterials;
      bones.push(o);
    } else {
      const m = meta.muscles[o.name];
      seg = m.segment;
      o.userData.mats = {
        xray: activeMaterial(MUSCLE_COLD, MUSCLE_HOT, 0.3),
        solid: new THREE.MeshStandardMaterial({
          color: 0x8f2f2a,
          emissive: 0xff8a1e,
          emissiveIntensity: 0,
          roughness: 0.7,
          side: THREE.DoubleSide,
        }),
      };
      act[`${m.group}_${m.side}`] = o.userData.mats;
      muscles.push(o);
    }
    (parts[seg] ??= { meshes: [], vis: 0 }).meshes.push(o);
  });
  for (const o of [...bones, ...muscles]) group.add(o);

  // rest geometry
  const R = meta.torso,
    rSM = V(R.shoulderL).add(V(R.shoulderR)).multiplyScalar(0.5),
    rHM = V(R.hipL).add(V(R.hipR)).multiplyScalar(0.5),
    rL = V(R.shoulderL).sub(V(R.shoulderR)),
    rU = rSM.clone().sub(rHM),
    rN = V().crossVectors(rL, rU).normalize(),
    rHipL = V(R.hipL).sub(V(R.hipR));
  const torsoRestInv = M().makeBasis(rL, rU, rN).invert();
  const pelvisRestInv = M()
    .makeBasis(rHipL, new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1))
    .invert();
  const rest = {};
  for (const [name, s] of Object.entries(meta.segments)) {
    const prox = V(s.prox),
      ar = V(s.dist).sub(prox);
    rest[name] = {
      prox,
      len: ar.length(),
      basisT: frameOf(ar.clone().normalize(), X, M()).transpose(),
    };
  }

  let stickLines;
  if (stick) {
    const g = new THREE.BufferGeometry();
    g.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(STICK.length * 6), 3),
    );
    const col = new Float32Array(STICK.length * 6);
    STICK.forEach(([a, b], i) => {
      // subject's left = green, right = magenta, midline = white
      const c =
        a % 2 === b % 2 ? (a % 2 ? [0.2, 1, 0.4] : [1, 0.2, 0.9]) : [1, 1, 1];
      col.set([...c, ...c], i * 6);
    });
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    stickLines = new THREE.LineSegments(
      g,
      new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false }),
    );
    stickLines.frustumCulled = false;
    stickLines.renderOrder = 10;
    group.add(stickLines);
  }

  const P = Array.from({ length: 33 }, () => V()),
    vis = new Float32Array(33);
  const lat = { arm_L: V(), arm_R: V(), leg_L: V(), leg_R: V() };
  let seen = false,
    k = 0;
  const q = new THREE.Quaternion(),
    T = M(),
    B = M(),
    S = M();
  const show = (seg, on) => {
    for (const m of parts[seg].meshes) m.visible = on && m.userData.layerOn;
  };
  const place = (seg, matrix) => {
    for (const m of parts[seg].meshes) m.matrix.copy(matrix);
  };
  // lateral reference of a two-bone limb: its flexion axis, or the trunk's left carried along when straight
  function limbLateral(key, a, b, down, left, sign) {
    const an = a.clone().normalize(),
      flex = V().crossVectors(a, b).multiplyScalar(sign);
    const sin = flex.length() / (a.length() * b.length() + 1e-6);
    const ref = left.clone().applyQuaternion(q.setFromUnitVectors(down, an));
    const w = THREE.MathUtils.smoothstep(sin, 0.15, 0.5);
    const out = ref.multiplyScalar(1 - w).addScaledVector(flex.normalize(), w);
    if (out.lengthSq() < 0.04)
      out.copy(left).applyQuaternion(q.setFromUnitVectors(down, an));
    out.normalize();
    const s = lat[key];
    if (s.lengthSq() === 0) s.copy(out);
    else s.lerp(out, 0.35).normalize();
    return s;
  }
  function bone(seg, p, d, lateral) {
    const r = rest[seg],
      a = d.clone().sub(p),
      len = a.length();
    if (len < 1e-3) return;
    frameOf(a.multiplyScalar(1 / len), lateral, B);
    T.makeTranslation(p)
      .multiply(B)
      // a limb cannot be much longer or shorter than the body's proportions: bad depth guesses stop here
      .multiply(
        S.makeScale(
          THREE.MathUtils.clamp(len / r.len, 0.6 * k, 1.3 * k),
          k,
          k,
        ),
      )
      .multiply(r.basisT)
      .multiply(M().makeTranslation(r.prox.clone().negate()));
    place(seg, T);
  }

  function update(lm, w, h) {
    if (!lm) {
      group.visible = false;
      seen = false;
      return false;
    }
    const kp = seen ? 0.5 : 1;
    lm.forEach((l, i) => {
      P[i].lerp(V([(l.x - 0.5) * w, (0.5 - l.y) * h, -l.z * w]), kp);
      vis[i] += ((l.visibility ?? 1) - vis[i]) * kp;
    });
    seen = true;
    const ok = (...ids) => Math.min(...ids.map((i) => vis[i])) > 0.5;
    if (!ok(11, 12)) {
      group.visible = false;
      return false;
    }
    group.visible = true;
    const sL = P[11],
      sR = P[12],
      sM = sL.clone().add(sR).multiplyScalar(0.5);
    const L = sL.clone().sub(sR);
    const hipsSeen = Math.min(vis[23], vis[24]) > 0.3;
    // hips out of frame (sitting at a desk): hang the trunk straight down from the shoulders
    let kNow = L.length() / rL.length();
    const hM = hipsSeen
      ? P[23].clone().add(P[24]).multiplyScalar(0.5)
      : sM.clone().add(new THREE.Vector3(0, -rU.length() * kNow, 0));
    const U = sM.clone().sub(hM);
    kNow = Math.max(kNow, U.length() / rU.length());
    k = k ? k + (kNow - k) * 0.3 : kNow;
    const Ld = L.clone().normalize(),
      Ud = U.clone().normalize(),
      N = V().crossVectors(L, U).normalize();
    // width and height never drop far below the body scale, so a sideways turn does not flatten the trunk
    const Lv = Ld.clone().multiplyScalar(
        Math.max(L.length(), 0.7 * k * rL.length()),
      ),
      Uv = Ud.clone().multiplyScalar(
        Math.max(U.length(), 0.7 * k * rU.length()),
      );
    T.makeTranslation(sM)
      .multiply(
        M()
          .makeBasis(Lv, Uv, N.clone().multiplyScalar(k))
          .multiply(torsoRestInv),
      )
      .multiply(M().makeTranslation(rSM.clone().negate()));
    place("torso", T);
    show("torso", true);
    const down = Ud.clone().negate();

    const pelvisOn = ok(23, 24);
    let hipLeft = Ld;
    if (pelvisOn) {
      const Lp = P[23].clone().sub(P[24]);
      hipLeft = Lp.clone().normalize();
      const Np = V().crossVectors(Lp, U).normalize();
      const len = THREE.MathUtils.clamp(
        Lp.length(),
        0.7 * k * rHipL.length(),
        1.4 * k * rHipL.length(),
      );
      T.makeTranslation(hM)
        .multiply(
          M()
            .makeBasis(
              hipLeft.clone().multiplyScalar(len),
              Ud.clone().multiplyScalar(k),
              Np.multiplyScalar(k),
            )
            .multiply(pelvisRestInv),
        )
        .multiply(M().makeTranslation(rHM.clone().negate()));
      place("pelvis", T);
    }
    show("pelvis", pelvisOn);

    ["L", "R"].forEach((s, i) => {
      const J = (n) => P[IDX[n][i]],
        id = (n) => IDX[n][i];
      const armOn = ok(id("shoulder"), id("elbow")),
        foreOn = armOn && ok(id("wrist"));
      if (armOn) {
        const u = J("elbow").clone().sub(J("shoulder")),
          f = J("wrist").clone().sub(J("elbow"));
        // elbow flexion axis = -(upper arm x forearm) corresponds to the model's +x at rest
        const l = foreOn
          ? limbLateral(`arm_${s}`, u, f, down, Ld, -1)
          : limbLateral(`arm_${s}`, u, u, down, Ld, -1);
        bone(`upperarm_${s}`, J("shoulder"), J("elbow"), l);
        if (foreOn) bone(`forearm_${s}`, J("elbow"), J("wrist"), l);
      }
      show(`upperarm_${s}`, armOn);
      show(`forearm_${s}`, foreOn);
      const thighOn = pelvisOn && ok(id("hip"), id("knee")),
        shinOn = thighOn && ok(id("ankle")),
        footOn = shinOn && ok(id("heel"), id("toe"));
      if (thighOn) {
        const t = J("knee").clone().sub(J("hip")),
          sh = J("ankle").clone().sub(J("knee"));
        const l = shinOn
          ? limbLateral(`leg_${s}`, t, sh, down, hipLeft, 1)
          : limbLateral(`leg_${s}`, t, t, down, hipLeft, 1);
        bone(`thigh_${s}`, J("hip"), J("knee"), l);
        if (shinOn) bone(`shin_${s}`, J("knee"), J("ankle"), l);
        if (footOn) bone(`foot_${s}`, J("heel"), J("toe"), l);
      }
      show(`thigh_${s}`, thighOn);
      show(`shin_${s}`, shinOn);
      show(`foot_${s}`, footOn);
    });

    if (stickLines) {
      const a = stickLines.geometry.attributes.position;
      STICK.forEach(([i, j], n) => {
        a.setXYZ(2 * n, P[i].x, P[i].y, P[i].z);
        a.setXYZ(2 * n + 1, P[j].x, P[j].y, P[j].z);
      });
      a.needsUpdate = true;
    }
    return true;
  }

  function setActivation(levels) {
    for (const [key, mats] of Object.entries(act)) {
      const v = levels?.[key] ?? 0;
      mats.xray.uniforms.act.value = v;
      mats.solid.emissiveIntensity = v * 1.6;
    }
  }
  return { group, bones, muscles, update, setActivation, points: P, vis };
}
