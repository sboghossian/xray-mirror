// node --test test-unit/
import test from "node:test";
import assert from "node:assert/strict";
import { createEstimator, GROUPS, SIDES } from "../public/activity.js";

// MediaPipe world space: metres, hip-centred, +y DOWN, subject faces the camera (-z),
// subject's left is +x. Index 0 of each pair is the subject's LEFT.
function standing() {
  const p = Array.from({ length: 33 }, () => [0, 0, 0]);
  p[0] = [0, -0.65, -0.1];
  for (const [k, s] of [
    [0, 1],
    [1, -1],
  ]) {
    p[11 + k] = [0.18 * s, -0.5, 0];
    p[13 + k] = [0.2 * s, -0.22, 0];
    p[15 + k] = [0.2 * s, 0.04, 0];
    p[23 + k] = [0.1 * s, 0, 0];
    p[25 + k] = [0.1 * s, 0.45, 0];
    p[27 + k] = [0.1 * s, 0.88, 0];
    p[29 + k] = [0.1 * s, 0.93, 0.05];
    p[31 + k] = [0.1 * s, 0.93, -0.15];
  }
  return p;
}
const toLm = (p, vis = () => 1) =>
  p.map(([x, y, z], i) => ({ x, y, z, visibility: vis(i) }));
// hold a pose (or play a function of time) and return the last estimate
function run(poseAt, seconds = 2, extras = () => ({}), vis) {
  const est = createEstimator();
  let r;
  for (let i = 0; i <= seconds * 30; i++) {
    const t = i / 30;
    r = est.update(
      toLm(typeof poseAt === "function" ? poseAt(t) : poseAt, vis),
      t,
      extras(t),
    );
  }
  return r;
}
const rot = (deg) => [
  Math.sin((deg * Math.PI) / 180),
  Math.cos((deg * Math.PI) / 180),
];

test("standing still: nothing works hard", () => {
  const { muscles } = run(standing());
  for (const g of GROUPS)
    for (const s of SIDES)
      assert.ok(
        muscles[`${g}_${s}`] < 0.25,
        `${g}_${s} = ${muscles[`${g}_${s}`]}`,
      );
});

function armOut(p, k, s, elevDeg) {
  const [sn, cs] = rot(elevDeg); // 0 = hanging, 90 = horizontal, out to the side
  const S = p[11 + k];
  p[13 + k] = [S[0] + s * 0.3 * sn, S[1] + 0.3 * cs, 0];
  p[15 + k] = [S[0] + s * 0.56 * sn, S[1] + 0.56 * cs, 0];
  return p;
}

test("left arm held out sideways: left deltoid, not right, not the pull-down muscles", () => {
  const { muscles } = run(armOut(standing(), 0, 1, 90));
  assert.ok(muscles.deltoid_L > 0.7, `deltoid_L ${muscles.deltoid_L}`);
  assert.ok(muscles.deltoid_R < 0.1, `deltoid_R ${muscles.deltoid_R}`);
  assert.ok(muscles.lats_L < 0.1, `lats_L ${muscles.lats_L}`);
});

test("lowering an arm slowly still loads the deltoid (eccentric), not its antagonist", () => {
  // 90 -> 45 degrees over 3 s, then read
  const { muscles } = run((t) => armOut(standing(), 0, 1, 90 - 15 * t), 3);
  assert.ok(muscles.deltoid_L > 0.4, `deltoid_L ${muscles.deltoid_L}`);
  assert.ok(muscles.lats_L < muscles.deltoid_L / 3, `lats_L ${muscles.lats_L}`);
});

test("forearm held level in front: biceps; forearm level behind a raised arm: triceps", () => {
  const p = standing();
  p[13] = [0.18, -0.2, 0];
  p[15] = [0.18, -0.2, -0.26]; // forward
  let { muscles } = run(p);
  assert.ok(muscles.biceps_L > 0.6, `biceps_L ${muscles.biceps_L}`);
  assert.ok(muscles.triceps_L < 0.1, `triceps_L ${muscles.triceps_L}`);
  const q = standing();
  q[13] = [0.18, -0.8, 0]; // upper arm straight up
  q[15] = [0.18, -0.8, 0.26]; // forearm pointing back
  ({ muscles } = run(q));
  assert.ok(muscles.triceps_L > 0.6, `triceps_L ${muscles.triceps_L}`);
  assert.ok(muscles.biceps_L < 0.2, `biceps_L ${muscles.biceps_L}`);
});

function squat() {
  const p = standing();
  const lean = rot(35);
  for (const [k, s] of [
    [0, 1],
    [1, -1],
  ]) {
    p[25 + k] = [0.12 * s, 0.03, -0.45]; // thighs level, knees forward
    p[27 + k] = [0.12 * s, 0.45, -0.3];
    p[29 + k] = [0.12 * s, 0.5, -0.25];
    p[31 + k] = [0.12 * s, 0.5, -0.45];
    p[11 + k] = [0.18 * s, -0.5 * lean[1], -0.5 * lean[0]];
    p[13 + k] = [0.2 * s, p[11 + k][1] + 0.28, p[11 + k][2]];
    p[15 + k] = [0.2 * s, p[11 + k][1] + 0.54, p[11 + k][2]];
  }
  return p;
}

test("squat hold: quads and glutes on both sides", () => {
  const { muscles } = run(squat());
  for (const s of SIDES) {
    assert.ok(
      muscles[`quads_${s}`] > 0.7,
      `quads_${s} ${muscles[`quads_${s}`]}`,
    );
    assert.ok(
      muscles[`glutes_${s}`] > 0.4,
      `glutes_${s} ${muscles[`glutes_${s}`]}`,
    );
  }
  assert.ok(muscles.hipflex_L < 0.15, `hipflex_L ${muscles.hipflex_L}`);
});

test("left knee raised: left hip flexors, right leg stabilises", () => {
  const p = standing();
  p[25] = [0.1, 0.02, -0.44];
  p[27] = [0.1, 0.44, -0.44];
  p[29] = [0.1, 0.49, -0.4];
  p[31] = [0.1, 0.49, -0.6];
  const { muscles, stance } = run(p);
  assert.ok(stance[0] < 0.1 && stance[1] > 0.9, `stance ${stance}`);
  assert.ok(muscles.hipflex_L > 0.6, `hipflex_L ${muscles.hipflex_L}`);
  assert.ok(muscles.hipflex_R < 0.1, `hipflex_R ${muscles.hipflex_R}`);
  assert.ok(muscles.glutes_R > 0.3, `glutes_R ${muscles.glutes_R}`);
});

test("heel raise: calves", () => {
  const p = standing();
  for (const k of [0, 1]) p[29 + k][1] = 0.84;
  const { muscles } = run(p);
  assert.ok(
    muscles.calves_L > 0.7 && muscles.calves_R > 0.7,
    `calves ${muscles.calves_L}`,
  );
});

test("hip hinge forward: back extensors and hamstrings, not the obliques", () => {
  const p = standing();
  const lean = rot(60);
  for (const [k, s] of [
    [0, 1],
    [1, -1],
  ]) {
    p[11 + k] = [0.18 * s, -0.5 * lean[1], -0.5 * lean[0]];
    p[13 + k] = [0.2 * s, p[11 + k][1] + 0.28, p[11 + k][2]];
    p[15 + k] = [0.2 * s, p[11 + k][1] + 0.54, p[11 + k][2]];
  }
  const { muscles } = run(p);
  assert.ok(muscles.erectors_L > 0.5, `erectors_L ${muscles.erectors_L}`);
  assert.ok(muscles.hamstrings_L > 0.4, `hamstrings_L ${muscles.hamstrings_L}`);
  assert.ok(muscles.obliques_L < 0.15, `obliques_L ${muscles.obliques_L}`);
});

test("brain map is contralateral: waving the right hand lights the left hemisphere", () => {
  const { brain } = run((t) => {
    const p = armOut(standing(), 1, -1, 90);
    p[16][1] += 0.15 * Math.sin(t * 12);
    return p;
  }, 3);
  assert.ok(brain.m1_L.hand > 0.3, `m1_L.hand ${brain.m1_L.hand}`);
  assert.ok(
    brain.m1_R.hand < brain.m1_L.hand / 3,
    `m1_R.hand ${brain.m1_R.hand}`,
  );
  assert.ok(brain.m1_L.leg < 0.15, `m1_L.leg ${brain.m1_L.leg}`);
});

test("talking lights the speech area and the face strip", () => {
  const { brain } = run(standing(), 3, (t) => ({
    jawOpen: 0.3 + 0.3 * Math.sin(t * 14),
  }));
  assert.ok(brain.broca > 0.4, `broca ${brain.broca}`);
  assert.ok(brain.m1_L.face > 0.3, `face ${brain.m1_L.face}`);
});

test("an arm the tracker cannot see reports nothing", () => {
  const hidden = new Set([11, 13, 15]);
  const { muscles } = run(
    armOut(standing(), 0, 1, 90),
    2,
    () => ({}),
    (i) => (hidden.has(i) ? 0.1 : 1),
  );
  assert.ok(muscles.deltoid_L < 0.02, `deltoid_L ${muscles.deltoid_L}`);
});

test("jittery landmarks do not make a standing person flicker", () => {
  let seed = 7;
  const rnd = () =>
    ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648 - 0.5) *
    0.02;
  const est = createEstimator();
  let peak = 0;
  for (let i = 0; i < 120; i++) {
    const p = standing().map(([x, y, z]) => [x + rnd(), y + rnd(), z + rnd()]);
    const { muscles } = est.update(toLm(p), i / 30);
    if (i > 30) for (const k in muscles) peak = Math.max(peak, muscles[k]);
  }
  assert.ok(peak < 0.35, `peak ${peak}`);
});

test("plank: arms and trunk carry the body, back extensors do not dominate", () => {
  // face down, head towards -z, body level 0.2 m above the hands and toes
  const p = Array.from({ length: 33 }, () => [0, 0, 0]);
  for (const [k, s] of [
    [0, -1],
    [1, 1],
  ]) {
    p[11 + k] = [0.18 * s, 0, -0.5];
    p[13 + k] = [0.2 * s, 0.14, -0.5];
    p[15 + k] = [0.2 * s, 0.28, -0.5];
    p[23 + k] = [0.1 * s, 0, 0];
    p[25 + k] = [0.1 * s, 0.03, 0.45];
    p[27 + k] = [0.1 * s, 0.08, 0.88];
    p[29 + k] = [0.1 * s, 0.02, 0.9];
    p[31 + k] = [0.1 * s, 0.2, 0.9];
  }
  const { muscles, support } = run(p);
  assert.ok(support > 0.8, `support ${support}`);
  assert.ok(muscles.triceps_L > 0.3, `triceps_L ${muscles.triceps_L}`);
  assert.ok(muscles.obliques_L > 0.4, `obliques_L ${muscles.obliques_L}`);
  assert.ok(muscles.erectors_L < 0.35, `erectors_L ${muscles.erectors_L}`);
});
