// Estimates muscle effort and a textbook brain map from body pose alone.
// Pure module: no DOM, no three.js. Input is MediaPipe pose WORLD landmarks
// (33 points, metres, origin at the hip centre, +y pointing DOWN, subject's left = "left").
//
// This is an estimate from movement, not a measurement. The model is quasi-static
// inverse dynamics: for each joint, the torque that gravity (plus the limb's own
// acceleration) puts on the segments beyond it, assigned to the muscle group that
// would have to resist it. It cannot see external loads, grip, or co-contraction,
// and a person sitting on a chair looks the same as a person holding a squat.

export const LM = {
  nose: 0,
  shoulder: [11, 12],
  elbow: [13, 14],
  wrist: [15, 16],
  hip: [23, 24],
  knee: [25, 26],
  ankle: [27, 28],
  heel: [29, 30],
  toe: [31, 32],
};
export const SIDES = ["L", "R"];
export const GROUPS = [
  "deltoid",
  "biceps",
  "triceps",
  "forearm",
  "pecs",
  "traps",
  "lats",
  "serratus",
  "obliques",
  "erectors",
  "glutes",
  "hipflex",
  "quads",
  "hamstrings",
  "adductors",
  "calves",
  "tibialis",
];
export const GROUP_LABEL = {
  deltoid: "Deltoid",
  biceps: "Biceps",
  triceps: "Triceps",
  forearm: "Forearm",
  pecs: "Chest",
  traps: "Trapezius",
  lats: "Teres major",
  serratus: "Serratus",
  obliques: "Obliques",
  erectors: "Back extensors",
  glutes: "Glutes",
  hipflex: "Hip flexors",
  quads: "Quadriceps",
  hamstrings: "Hamstrings",
  adductors: "Adductors",
  calves: "Calves",
  tibialis: "Shin",
};

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const norm = (a) => {
  const l = len(a);
  return l > 1e-9 ? mul(a, 1 / l) : [0, 0, 0];
};
const clamp = (v, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
const smooth = (v, a, b) => {
  const t = clamp((v - a) / (b - a));
  return t * t * (3 - 2 * t);
};
const deg = (r) => (r * 180) / Math.PI;
const angle = (a, b) => Math.acos(clamp(dot(norm(a), norm(b)), -1, 1));
const DOWN = [0, 1, 0];
const UP = [0, -1, 0];
// part of v perpendicular to unit vector n
const perp = (v, n) => sub(v, mul(n, dot(v, n)));

// One Euro filter (Casiez et al. 2012) for one scalar.
class OneEuro {
  constructor(minCutoff = 1.5, beta = 0.4, dCutoff = 1) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.x = null;
    this.dx = 0;
  }
  static alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }
  filter(x, dt) {
    if (this.x === null) {
      this.x = x;
      return x;
    }
    const dx = (x - this.x) / dt;
    this.dx += OneEuro.alpha(this.dCutoff, dt) * (dx - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += OneEuro.alpha(cutoff, dt) * (x - this.x);
    return this.x;
  }
}

export function createEstimator() {
  const filters = Array.from({ length: 33 }, () => [
    new OneEuro(),
    new OneEuro(),
    new OneEuro(),
  ]);
  let prevT = null;
  let prevP = null;
  let vel = Array.from({ length: 33 }, () => [0, 0, 0]);
  let acc = Array.from({ length: 33 }, () => [0, 0, 0]);
  const out = {};
  for (const g of GROUPS) for (const s of SIDES) out[`${g}_${s}`] = 0;
  const brainOut = {};
  let prevJaw = null;
  let jawSpeed = 0;

  function update(world, t, extras = {}) {
    const dt = prevT === null ? 1 / 30 : clamp(t - prevT, 1 / 240, 0.25);
    prevT = t;
    const P = world.map((p, i) => [
      filters[i][0].filter(p.x, dt),
      filters[i][1].filter(p.y, dt),
      filters[i][2].filter(p.z, dt),
    ]);
    const vis = world.map((p) => p.visibility ?? 1);
    if (prevP) {
      const kv = clamp(dt / 0.08),
        ka = clamp(dt / 0.15);
      for (let i = 0; i < 33; i++) {
        const v = mul(sub(P[i], prevP[i]), 1 / dt);
        const nv = add(vel[i], mul(sub(v, vel[i]), kv));
        const a = mul(sub(nv, vel[i]), 1 / dt);
        vel[i] = nv;
        acc[i] = add(acc[i], mul(sub(a, acc[i]), ka));
        const al = len(acc[i]);
        if (al > 15) acc[i] = mul(acc[i], 15 / al);
      }
    }
    prevP = P;
    // gravity as felt by a point: g minus its own acceleration, in units of g
    const gAt = (i) => sub(DOWN, mul(acc[i], 1 / 9.81));
    const gMid = (i, j) => mul(add(gAt(i), gAt(j)), 0.5);

    const sh = [P[11], P[12]],
      hp = [P[23], P[24]];
    const shMid = mul(add(sh[0], sh[1]), 0.5),
      hpMid = mul(add(hp[0], hp[1]), 0.5);
    const trunk = sub(shMid, hpMid);
    const trunkU = norm(trunk);
    const leftDir = norm(sub(sh[0], sh[1]));
    const chest = norm(cross(leftDir, trunkU)); // the way the chest faces
    // "forward" along the floor; when the trunk is level the chest faces the floor or the ceiling
    const chestH = perp(chest, DOWN);
    const forward = len(chestH) > 0.15 ? norm(chestH) : chest;
    const trunkLean = len(perp(trunkU, DOWN)); // 0 upright, 1 horizontal
    const leanDir = norm(perp(trunkU, DOWN));
    const fwdness =
      len(chestH) > 0.15 ? dot(leanDir, forward) : Math.sign(dot(chest, DOWN));
    const leanFwd = trunkLean * clamp(fwdness);
    const leanBack = trunkLean * clamp(-fwdness);
    const leanLeft = trunkLean * dot(leanDir, leftDir); // + = leaning to subject's left
    // twist: shoulder line against hip line, seen from above
    const hipLeft = norm(perp(sub(hp[0], hp[1]), trunkU));
    const shLeft = norm(perp(sub(sh[0], sh[1]), trunkU));
    const twist = clamp(angle(hipLeft, shLeft) / (Math.PI / 4));

    // arms supporting the body (plank, push-up): trunk and legs near horizontal,
    // hands below the shoulders. A standing forward fold has vertical legs, so it does not count.
    const handsBelow = clamp(
      ((P[15][1] + P[16][1]) / 2 - shMid[1]) / (0.6 * len(trunk) + 1e-6),
    );
    const legLine = norm(sub(mul(add(P[27], P[28]), 0.5), hpMid));
    const support =
      smooth(trunkLean, 0.7, 0.9) *
      smooth(handsBelow, 0.3, 0.7) *
      smooth(len(perp(legLine, DOWN)), 0.6, 0.85);

    // which feet carry weight: the lowest ankle, and any ankle within a few cm of it
    const floorY = Math.max(P[27][1], P[28][1]);
    const stance = [27, 28].map(
      (i) => 1 - smooth(floorY - P[i][1], 0.05, 0.15),
    );
    const stanceSum = stance[0] + stance[1] || 1;
    const loadShare = stance.map((s) => (s / stanceSum) * (1 - support * 0.6));

    const raw = {};
    const joints = { trunkLean: deg(Math.asin(clamp(trunkLean))) };
    const speed = {};

    SIDES.forEach((side, k) => {
      const S = P[LM.shoulder[k]],
        E = P[LM.elbow[k]],
        W = P[LM.wrist[k]];
      const H = P[LM.hip[k]],
        K = P[LM.knee[k]],
        A = P[LM.ankle[k]];
      const heel = P[LM.heel[k]],
        toe = P[LM.toe[k]];
      const armOk = Math.min(
        vis[LM.shoulder[k]],
        vis[LM.elbow[k]],
        vis[LM.wrist[k]],
      );
      const legOk = Math.min(vis[LM.hip[k]], vis[LM.knee[k]], vis[LM.ankle[k]]);
      const aGate = smooth(armOk, 0.3, 0.6),
        lGate = smooth(legOk, 0.3, 0.6);
      const sideSign = k === 0 ? 1 : -1; // +1 = subject's left
      const lateral = mul(leftDir, sideSign); // pointing away from the midline on this side

      // ----- shoulder -----
      const u = sub(E, S),
        f = sub(W, E);
      const Lu = len(u),
        Lf = len(f);
      const mU = 0.028,
        mF = 0.022; // segment mass / body mass (upper arm, forearm + hand)
      const gU = gMid(LM.shoulder[k], LM.elbow[k]),
        gF = gMid(LM.elbow[k], LM.wrist[k]);
      const tau = add(
        mul(cross(mul(u, 0.5), gU), mU),
        mul(cross(add(u, mul(f, 0.5)), gF), mF),
      );
      const tauMax = mU * 0.5 * Lu + mF * (Lu + 0.5 * Lf) + 1e-9;
      const tauN = clamp(len(tau) / tauMax, 0, 1.5);
      const armCom = add(mul(u, 0.5 * mU), mul(add(u, mul(f, 0.5)), mF));
      const armU = norm(armCom);
      const gArm = mul(add(mul(gU, mU), mul(gF, mF)), 1 / (mU + mF));
      const resist = norm(mul(perp(gArm, armU), -1)); // direction the shoulder muscles must pull the arm
      const lift = clamp(dot(resist, UP));
      const pullDown = clamp(dot(resist, DOWN));
      const pullIn = clamp(-dot(resist, lateral));
      const armFwd = clamp(dot(norm(perp(armU, DOWN)), forward));
      const armBack = clamp(-dot(norm(perp(armU, DOWN)), forward));
      const elev = angle(u, DOWN); // 0 = hanging, PI = straight up
      const overhead = smooth(deg(elev), 80, 150);

      // ----- elbow -----
      const fU = norm(f);
      const pFlex = norm(perp(mul(u, -1), fU)); // the way the wrist moves when the elbow flexes
      const gFlex = dot(gF, pFlex); // > 0: gravity flexes the elbow, extensors resist
      const elbowFlex = angle(u, f);

      const pushUp =
        support * (0.4 + 0.5 * Math.sin(clamp(elbowFlex, 0, Math.PI / 2)));
      const wristSpeed = len(sub(vel[LM.wrist[k]], vel[LM.elbow[k]]));

      raw[`deltoid_${side}`] = aGate * (tauN * lift + support * 0.4);
      raw[`traps_${side}`] = aGate * (0.45 * tauN * lift + 0.5 * overhead);
      raw[`pecs_${side}`] =
        aGate *
        (0.45 * tauN * lift * armFwd + 0.9 * tauN * pullIn + support * 0.6);
      raw[`lats_${side}`] =
        aGate * (tauN * pullDown + 0.5 * tauN * lift * armBack);
      raw[`serratus_${side}`] =
        aGate * (0.4 * tauN * lift * armFwd + 0.4 * overhead + support * 0.4);
      raw[`biceps_${side}`] =
        aGate * (0.9 * clamp(-gFlex) + 0.25 * tauN * lift * armFwd);
      raw[`triceps_${side}`] = aGate * (0.9 * clamp(gFlex) + pushUp);
      raw[`forearm_${side}`] =
        aGate *
        (0.5 * len(cross(fU, gF)) +
          0.5 * clamp(wristSpeed / 1.5) +
          support * 0.3);

      // ----- trunk (per side) -----
      const sideLean = clamp(-leanLeft * sideSign); // leaning away from this side loads it
      // the trunk's lean and twist need both shoulders and both hips in view
      const tGate = smooth(Math.min(vis[11], vis[12], vis[23], vis[24]), 0.3, 0.6);
      raw[`erectors_${side}`] =
        tGate *
        ((1 - support) * (0.08 + 0.9 * leanFwd + 0.5 * sideLean) +
          support * 0.2);
      raw[`obliques_${side}`] =
        tGate *
        (0.9 * leanBack + 0.6 * sideLean + 0.6 * twist + support * 0.6);

      // ----- hip, knee, ankle -----
      const th = sub(K, H),
        shn = sub(A, K);
      const thU = norm(th),
        shU = norm(shn);
      const thighHoriz = len(perp(thU, DOWN));
      const kneeFlex = angle(th, shn);
      const bent = smooth(deg(kneeFlex), 5, 25);
      const st = stance[k],
        load = clamp(loadShare[k] * 2, 0, 2);
      const swing = 1 - st;
      const gShin = gMid(LM.knee[k], LM.ankle[k]);
      const pKnee = norm(perp(mul(th, -1), shU));
      const gKnee = dot(gShin, pKnee); // > 0: gravity flexes the knee, quads resist
      const thighFwd = clamp(dot(norm(perp(thU, DOWN)), forward));
      const thighBack = clamp(-dot(norm(perp(thU, DOWN)), forward));
      const otherSwing = 1 - stance[1 - k];
      const footLen = len(sub(toe, heel)) + 1e-6;
      const heelRise = clamp((toe[1] - heel[1]) / (0.45 * footLen)); // heel above toe
      const toeRise = clamp((heel[1] - toe[1]) / (0.35 * footLen));
      const stanceWidth = clamp(
        (len(perp(sub(P[27], P[28]), DOWN)) / (len(sub(hp[0], hp[1])) + 1e-6) -
          1.5) /
          1.5,
      );
      const upright = 1 - support;

      raw[`quads_${side}`] =
        lGate *
        (upright * load * thighHoriz * bent +
          swing * 0.8 * clamp(gKnee) +
          support * 0.3);
      raw[`glutes_${side}`] =
        lGate *
        (upright * load * (0.6 * thighHoriz * bent + 0.6 * leanFwd) +
          st * otherSwing * 0.5 +
          swing * 0.6 * thighHoriz * thighBack +
          support * 0.3);
      raw[`hamstrings_${side}`] =
        lGate *
        (upright * load * (0.7 * leanFwd + 0.3 * thighHoriz * bent) +
          swing * (0.8 * clamp(-gKnee) + 0.4 * thighHoriz * thighBack));
      raw[`hipflex_${side}`] = lGate * swing * thighHoriz * thighFwd;
      raw[`adductors_${side}`] =
        lGate * upright * load * (0.35 * thighHoriz * bent + 0.4 * stanceWidth);
      raw[`calves_${side}`] =
        lGate *
        upright *
        load *
        (0.1 + 0.9 * heelRise + 0.15 * thighHoriz * bent);
      raw[`tibialis_${side}`] =
        lGate * (swing * 0.6 * toeRise + upright * load * 0.5 * leanBack);

      joints[`shoulder_${side}`] = deg(elev);
      joints[`elbow_${side}`] = deg(elbowFlex);
      joints[`hip_${side}`] = deg(angle(th, mul(trunk, -1)));
      joints[`knee_${side}`] = deg(kneeFlex);
      speed[`hand_${side}`] =
        aGate * len(sub(vel[LM.wrist[k]], vel[LM.shoulder[k]]));
      speed[`arm_${side}`] =
        aGate * len(sub(vel[LM.elbow[k]], vel[LM.shoulder[k]]));
      speed[`leg_${side}`] =
        lGate *
        Math.max(
          len(sub(vel[LM.ankle[k]], vel[LM.hip[k]])),
          len(sub(vel[LM.knee[k]], vel[LM.hip[k]])),
        );
    });
    const trunkSpeed = len(
      sub(mul(add(vel[11], vel[12]), 0.5), mul(add(vel[23], vel[24]), 0.5)),
    );

    // smooth: fast attack, slower release
    for (const key in raw) {
      const target = clamp(raw[key]);
      const tauS = target > out[key] ? 0.08 : 0.25;
      out[key] += clamp(dt / tauS) * (target - out[key]);
    }

    // ----- textbook brain map (illustrative) -----
    const jaw = extras.jawOpen ?? 0;
    if (prevJaw !== null)
      jawSpeed += clamp(dt / 0.2) * (Math.abs(jaw - prevJaw) / dt - jawSpeed);
    prevJaw = jaw;
    const mean = (keys) => keys.reduce((s, x) => s + out[x], 0) / keys.length;
    const part = (side) => ({
      leg: clamp(
        0.6 * clamp(speed[`leg_${side}`] / 1.2) +
          0.6 *
            mean(
              ["glutes", "quads", "hamstrings", "calves", "hipflex"].map(
                (g) => `${g}_${side}`,
              ),
            ),
      ),
      trunk: clamp(
        0.5 * clamp(trunkSpeed / 0.6) +
          0.7 * mean(["erectors", "obliques"].map((g) => `${g}_${side}`)),
      ),
      arm: clamp(
        0.6 * clamp(speed[`arm_${side}`] / 1.2) +
          0.7 *
            mean(["deltoid", "biceps", "triceps"].map((g) => `${g}_${side}`)),
      ),
      hand: clamp(
        0.8 * clamp(speed[`hand_${side}`] / 1.5) +
          0.5 * out[`forearm_${side}`] +
          (extras[`hand_${side}`] ?? 0),
      ),
      face: clamp(0.25 * jaw + clamp(jawSpeed / 2) + (extras.faceMotion ?? 0)),
    });
    // the left body is driven by the right hemisphere and the other way round
    const target = {
      m1_R: part("L"),
      m1_L: part("R"),
    };
    const scale = (o, s) =>
      Object.fromEntries(Object.entries(o).map(([k, v]) => [k, v * s]));
    target.s1_R = scale(target.m1_R, 0.7);
    target.s1_L = scale(target.m1_L, 0.7);
    const total = clamp(
      (speed.hand_L + speed.hand_R + speed.leg_L + speed.leg_R) / 3 +
        trunkSpeed,
    );
    const oneLeg = Math.abs(stance[0] - stance[1]);
    target.cerebellum = clamp(0.15 + 0.7 * total + 0.4 * oneLeg);
    target.visual = 0.6;
    target.broca = clamp(jawSpeed / 1.5);
    target.sma = clamp(
      SIDES.reduce(
        (s, _, k) => s + len(acc[LM.wrist[k]]) + len(acc[LM.ankle[k]]),
        0,
      ) / 20,
    );
    const kb = clamp(dt / 0.2);
    const blend = (cur, tgt) => (cur ?? 0) + kb * (tgt - (cur ?? 0));
    for (const key in target) {
      if (typeof target[key] === "number")
        brainOut[key] = blend(brainOut[key], target[key]);
      else {
        brainOut[key] ??= {};
        for (const p in target[key])
          brainOut[key][p] = blend(brainOut[key][p], target[key][p]);
      }
    }
    return { muscles: out, brain: brainOut, joints, stance, support };
  }
  return { update };
}
