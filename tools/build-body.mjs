// Builds public/models/body.glb and body-meta.json: full skeleton and muscle groups,
// one node per body segment (bones) and per muscle group per side.
// Run from the project root:  node tools/build-body.mjs
// Source meshes: BodyParts3D, (c) The Database Center for Life Science, CC BY-SA 2.1 Japan.
import fs from "fs";
import { Document, NodeIO } from "@gltf-transform/core";
import { MeshoptSimplifier } from "meshoptimizer";

const DIRS = [
  "raw/obj/partof_BP3D_4.0_obj_99/",
  "raw/obj-isa/isa_BP3D_4.0_obj_99/",
];
// concept name -> element file ids (is-a tree)
const concepts = new Map();
for (const l of fs
  .readFileSync("raw/isa_element_parts.txt", "utf8")
  .trim()
  .split("\n")
  .slice(1)) {
  const [, name, id] = l.split("\t");
  const k = name.toLowerCase();
  if (!concepts.has(k)) concepts.set(k, new Set());
  concepts.get(k).add(id.trim());
}
const ids = (name) => {
  const s = concepts.get(name.toLowerCase());
  if (!s) throw new Error(`no concept "${name}"`);
  return [...s];
};
const idsMatching = (re) => {
  const out = new Set();
  for (const [name, set] of concepts)
    if (re.test(name)) for (const id of set) out.add(id);
  return [...out];
};

function loadObj(id) {
  const file = DIRS.map((d) => d + id + ".obj").find((f) => fs.existsSync(f));
  if (!file) throw new Error(`no OBJ for ${id}`);
  const pos = [],
    idx = [];
  for (const line of fs.readFileSync(file, "utf8").split("\n")) {
    if (line.startsWith("v ")) {
      const p = line.split(/\s+/);
      // body frame: mm, +x = subject's left, +y = up, +z = anterior
      pos.push(+p[1], +p[3], -p[2]);
    } else if (line.startsWith("f ")) {
      const p = line
        .trim()
        .split(/\s+/)
        .slice(1)
        .map((t) => parseInt(t, 10) - 1);
      for (let i = 1; i + 1 < p.length; i++) idx.push(p[0], p[i], p[i + 1]);
    }
  }
  return { pos, idx };
}
const cache = new Map();
const part = (id) => {
  if (!cache.has(id)) cache.set(id, loadObj(id));
  return cache.get(id);
};
function merge(list) {
  const pos = [],
    idx = [];
  for (const id of [...new Set(list)]) {
    const m = part(id),
      base = pos.length / 3;
    for (const v of m.pos) pos.push(v);
    for (const i of m.idx) idx.push(i + base);
  }
  return { pos: new Float32Array(pos), idx: new Uint32Array(idx) };
}
function simplify({ pos, idx }, targetTris) {
  // weld
  const map = new Map(),
    remap = new Uint32Array(pos.length / 3),
    wp = [];
  for (let i = 0; i < pos.length / 3; i++) {
    const k = `${pos[3 * i].toFixed(2)},${pos[3 * i + 1].toFixed(2)},${pos[3 * i + 2].toFixed(2)}`;
    let j = map.get(k);
    if (j === undefined) {
      j = wp.length / 3;
      map.set(k, j);
      wp.push(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    }
    remap[i] = j;
  }
  let p = new Float32Array(wp),
    ix = idx.map((i) => remap[i]);
  if (ix.length / 3 > targetTris)
    [ix] = MeshoptSimplifier.simplify(ix, p, 3, targetTris * 3, 0.03, []);
  // compact
  const used = new Int32Array(p.length / 3).fill(-1),
    op = [],
    oi = new Uint32Array(ix.length);
  for (let k = 0; k < ix.length; k++) {
    const i = ix[k];
    if (used[i] < 0) {
      used[i] = op.length / 3;
      op.push(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
    }
    oi[k] = used[i];
  }
  return { pos: new Float32Array(op), idx: oi };
}
const verts = (m) => {
  const a = [];
  for (let i = 0; i < m.pos.length; i += 3)
    a.push([m.pos[i], m.pos[i + 1], m.pos[i + 2]]);
  return a;
};
const mean = (pts) =>
  pts
    .reduce((s, p) => [s[0] + p[0], s[1] + p[1], s[2] + p[2]], [0, 0, 0])
    .map((v) => v / pts.length);
// mean of the fraction of vertices that score highest
const extreme = (m, score, frac = 0.03) => {
  const v = verts(m).sort((a, b) => score(b) - score(a));
  return mean(v.slice(0, Math.max(1, Math.ceil(v.length * frac))));
};
const centroidX = (id) => {
  const p = part(id).pos;
  let s = 0;
  for (let i = 0; i < p.length; i += 3) s += p[i];
  return s / (p.length / 3);
};
// split an unsided concept by which side of the body each element sits on
const bySide = (list, side) =>
  list.filter((id) => (side === "L" ? centroidX(id) > 0 : centroidX(id) < 0));

const SIDES = { L: "left", R: "right" };
const bones = {},
  boneTarget = {};
const segments = {};
const sided = (name, side) => ids(`${SIDES[side]} ${name}`);

bones.torso = [
  ...ids("rib"),
  ...ids("costal cartilage"),
  ...ids("thoracic vertebra"),
  ...ids("lumbar vertebra"),
  ...idsMatching(/^(manubrium|body of sternum|xiphoid process)$/),
  ...["L", "R"].flatMap((s) => [
    ...sided("clavicle", s),
    ...sided("scapula", s),
  ]),
];
boneTarget.torso = 36000;
bones.pelvis = [
  ...ids("sacrum"),
  ...["L", "R"].flatMap((s) => sided("hip bone", s)),
];
boneTarget.pelvis = 9000;
const up = (p) => p[1],
  down = (p) => -p[1];
const torsoRest = {};
for (const s of ["L", "R"]) {
  const sign = s === "L" ? 1 : -1;
  const humerus = merge(sided("humerus", s)),
    femur = merge(sided("femur", s));
  const radius = merge(sided("radius", s)),
    ulna = merge(sided("ulna", s));
  const tibia = merge(sided("tibia", s));
  const foot = [
    ...bySide(ids("tarsal bone"), s),
    ...bySide(ids("metatarsal bone"), s),
    ...bySide(ids("phalanx of toe"), s),
  ];
  const footMesh = merge(foot);
  const shoulder = extreme(humerus, up, 0.04);
  const elbow = extreme(humerus, down, 0.04);
  const wrist = mean([extreme(radius, down, 0.04), extreme(ulna, down, 0.04)]);
  // hip joint = centre of the femoral head: the top of the femur nearest the midline
  const hip = extreme(femur, (p) => p[1] - sign * p[0] * 0.8, 0.05);
  const knee = extreme(femur, down, 0.04);
  const ankle = extreme(tibia, down, 0.04);
  const heel = extreme(footMesh, (p) => -p[2], 0.03);
  const toe = extreme(footMesh, (p) => p[2], 0.02);
  bones[`upperarm_${s}`] = sided("humerus", s);
  bones[`forearm_${s}`] = [...sided("radius", s), ...sided("ulna", s)];
  bones[`thigh_${s}`] = [...sided("femur", s), ...sided("patella", s)];
  bones[`shin_${s}`] = [...sided("tibia", s), ...sided("fibula", s)];
  bones[`foot_${s}`] = foot;
  for (const k of ["upperarm", "forearm", "thigh", "shin", "foot"])
    boneTarget[`${k}_${s}`] = 3000;
  segments[`upperarm_${s}`] = { prox: shoulder, dist: elbow };
  segments[`forearm_${s}`] = { prox: elbow, dist: wrist };
  segments[`thigh_${s}`] = { prox: hip, dist: knee };
  segments[`shin_${s}`] = { prox: knee, dist: ankle };
  segments[`foot_${s}`] = { prox: heel, dist: toe };
  torsoRest[`shoulder${s}`] = shoulder;
  torsoRest[`hip${s}`] = hip;
}

// muscle groups: every sided element whose concept name matches
const GROUPS = {
  deltoid: [/ deltoid$/, "upperarm"],
  biceps: [/biceps brachii$|^(left|right) brachialis$/, "upperarm"],
  triceps: [/triceps brachii$/, "upperarm"],
  forearm: [
    /^(left|right) (brachioradialis|flexor carpi (radialis|ulnaris)|extensor carpi (radialis longus|radialis brevis|ulnaris)|extensor digitorum|flexor digitorum (superficialis|profundus)|pronator teres|palmaris longus)$/,
    "forearm",
  ],
  pecs: [/pectoralis major$/, "torso"],
  traps: [/part of (left|right) trapezius$/, "torso"],
  lats: [/^(left|right) teres major$/, "torso"],
  serratus: [/^(left|right) serratus anterior$/, "torso"],
  obliques: [/^(left|right) external oblique$/, "torso"],
  erectors: [/^(left|right) (iliocostalis|longissimus|spinalis) /, "torso"],
  glutes: [/^(left|right) gluteus (maximus|medius)$/, "pelvis"],
  hipflex: [
    /^(left|right) (iliacus|psoas major|sartorius|tensor fasciae latae)$/,
    "thigh",
  ],
  quads: [
    /^(left|right) (rectus femoris|vastus (lateralis|medialis|intermedius))$/,
    "thigh",
  ],
  hamstrings: [
    /biceps femoris$|^(left|right) (semitendinosus|semimembranosus)$/,
    "thigh",
  ],
  adductors: [
    /^(left|right) (adductor (magnus|longus|brevis)|gracilis)$/,
    "thigh",
  ],
  calves: [/gastrocnemius$|^(left|right) soleus$/, "shin"],
  tibialis: [
    /^(left|right) (tibialis anterior|extensor digitorum longus)$/,
    "shin",
  ],
};
const SKIP = /artery|vein|nerve|tendon|branch|bursa|fascia|aponeurosis|sheath/;
const muscles = {},
  muscleMeta = {};
for (const [group, [re, seg]] of Object.entries(GROUPS))
  for (const s of ["L", "R"]) {
    const sideRe = new RegExp(`(^| )${SIDES[s]} `);
    const list = idsMatching({
      test: (n) => sideRe.test(n) && !SKIP.test(n) && re.test(n),
    });
    if (!list.length) throw new Error(`no meshes for ${group} ${s}`);
    const name = `m_${group}_${s}`;
    muscles[name] = list;
    const block = seg === "torso" || seg === "pelvis";
    muscleMeta[name] = {
      group,
      side: s,
      segment: block ? seg : `${seg}_${s}`,
    };
  }

await MeshoptSimplifier.ready;
const doc = new Document(),
  buf = doc.createBuffer(),
  scene = doc.createScene();
let tris = 0;
// Forearm and shin muscles end in long tendons that run into the hand and foot, which a
// stretched segment turns into spikes, and they start above the elbow / knee, which pokes out
// behind a bent joint: keep only the part between the two joints.
function clipPast({ pos, idx }, { prox, dist }, lo = -0.08, hi = 1.04) {
  const ax = dist.map((v, i) => v - prox[i]),
    l2 = ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2;
  const past = (i) => {
    const t =
      ((pos[3 * i] - prox[0]) * ax[0] +
        (pos[3 * i + 1] - prox[1]) * ax[1] +
        (pos[3 * i + 2] - prox[2]) * ax[2]) /
      l2;
    return t < lo || t > hi;
  };
  const keep = [];
  for (let i = 0; i < idx.length; i += 3)
    if (!past(idx[i]) && !past(idx[i + 1]) && !past(idx[i + 2]))
      keep.push(idx[i], idx[i + 1], idx[i + 2]);
  return { pos, idx: new Uint32Array(keep) };
}
function addNode(name, list, target, clip) {
  let raw = merge(list);
  if (clip) raw = clipPast(raw, clip);
  const m = simplify(raw, target);
  const prim = doc
    .createPrimitive()
    .setAttribute(
      "POSITION",
      doc.createAccessor().setType("VEC3").setArray(m.pos).setBuffer(buf),
    )
    .setIndices(
      doc
        .createAccessor()
        .setType("SCALAR")
        .setArray(m.pos.length / 3 > 65535 ? m.idx : new Uint16Array(m.idx))
        .setBuffer(buf),
    );
  scene.addChild(
    doc.createNode(name).setMesh(doc.createMesh(name).addPrimitive(prim)),
  );
  tris += m.idx.length / 3;
  console.log(
    " ",
    name.padEnd(18),
    String(list.length).padStart(3),
    "parts",
    raw.idx.length / 3,
    "->",
    m.idx.length / 3,
  );
}
for (const [seg, list] of Object.entries(bones))
  addNode(`b_${seg}`, list, boneTarget[seg]);
for (const [name, list] of Object.entries(muscles)) {
  const { group, segment } = muscleMeta[name];
  const tendons = ["forearm", "calves", "tibialis"].includes(group);
  addNode(name, list, 2600, tendons && segments[segment]);
}
doc.getRoot().getAsset().copyright =
  "BodyParts3D, (c) The Database Center for Life Science, CC BY-SA 2.1 Japan (decimated)";
await new NodeIO().write("public/models/body.glb", doc);
console.log(
  "public/models/body.glb",
  (fs.statSync("public/models/body.glb").size / 1e6).toFixed(2),
  "MB",
  tris,
  "tris",
);

const r = (a) => a.map((v) => +v.toFixed(1));
const rr = (o) =>
  Object.fromEntries(Object.entries(o).map(([k, v]) => [k, r(v)]));
fs.writeFileSync(
  "public/models/body-meta.json",
  JSON.stringify(
    {
      attribution:
        "BodyParts3D, © The Database Center for Life Science, licensed under CC Attribution-Share Alike 2.1 Japan",
      units: "mm; +x subject's left, +y up, +z anterior",
      torso: rr(torsoRest),
      segments: Object.fromEntries(
        Object.entries(segments).map(([k, v]) => [k, rr(v)]),
      ),
      muscles: muscleMeta,
    },
    null,
    1,
  ),
);
const d = (a, b) =>
  Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]).toFixed(0);
console.log(
  "rest (mm): shoulder width",
  d(torsoRest.shoulderL, torsoRest.shoulderR),
  "hip width",
  d(torsoRest.hipL, torsoRest.hipR),
  "upper arm",
  d(segments.upperarm_L.prox, segments.upperarm_L.dist),
  "forearm",
  d(segments.forearm_L.prox, segments.forearm_L.dist),
  "thigh",
  d(segments.thigh_L.prox, segments.thigh_L.dist),
  "shin",
  d(segments.shin_L.prox, segments.shin_L.dist),
  "foot",
  d(segments.foot_L.prox, segments.foot_L.dist),
);
